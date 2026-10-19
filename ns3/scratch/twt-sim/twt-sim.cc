/*
 * 802.11ax TWT vs legacy power save vs always-on: energy / throughput / latency / loss.
 *
 * One HE AP, nSta HE stations with uplink UDP traffic towards the AP.
 *   --mode=cam : no power save (radio always on)
 *   --mode=psm : legacy power save (ns-3 DefaultPowerSaveManager, PS-Poll, wake for beacons)
 *   --mode=twt : individual TWT (TwtPowerSaveManager in this folder)
 *
 * Writes one CSV row per station to --outFile (see README.md of the sweep scripts):
 *   staId,energyJ,txPackets,rxPackets,rxBytes,meanDelayMs,p95DelayMs,sleepFraction
 *
 * Measurement window: [warmup, warmup + simTime]. Traffic is generated only in that
 * window; energy and sleep time are measured over it; the simulation then runs a
 * drain period so packets still queued at the end can be delivered (packets never
 * delivered count as lost).
 *
 * Example:
 *   ./ns3 run "twt-sim --mode=twt --nSta=20 --wakeIntervalMs=100 --spDurationMs=10"
 */

#include "twt-power-save-manager.h"

#include "ns3/ap-wifi-mac.h"
#include "ns3/basic-energy-source-helper.h"
#include "ns3/boolean.h"
#include "ns3/command-line.h"
#include "ns3/config.h"
#include "ns3/double.h"
#include "ns3/flow-monitor-helper.h"
#include "ns3/histogram.h"
#include "ns3/internet-stack-helper.h"
#include "ns3/ipv4-address-helper.h"
#include "ns3/ipv4-flow-classifier.h"
#include "ns3/log.h"
#include "ns3/mobility-helper.h"
#include "ns3/neighbor-cache-helper.h"
#include "ns3/packet-sink-helper.h"
#include "ns3/pointer.h"
#include "ns3/random-variable-stream.h"
#include "ns3/rng-seed-manager.h"
#include "ns3/sta-wifi-mac.h"
#include "ns3/string.h"
#include "ns3/udp-client-server-helper.h"
#include "ns3/uinteger.h"
#include "ns3/wifi-net-device.h"
#include "ns3/wifi-phy-state-helper.h"
#include "ns3/wifi-radio-energy-model-helper.h"
#include "ns3/wifi-static-setup-helper.h"
#include "ns3/yans-wifi-helper.h"

#include <fstream>
#include <iomanip>
#include <map>

using namespace ns3;

NS_LOG_COMPONENT_DEFINE("TwtSim");

namespace
{

/// Tracks time spent in SLEEP by one PHY within a measurement window.
struct SleepTracker
{
    Time winStart;
    Time winEnd;
    Time sleep{0};
    Time lastEnd{0};          // end of the last state period reported by the trace
    WifiPhyState lastState{WifiPhyState::IDLE};
    bool closed{false};

    void Add(Time start, Time duration, WifiPhyState state)
    {
        if (closed)
        {
            return; // the state in progress at the window end was accounted in Close()
        }
        const Time end = start + duration;
        lastEnd = end;
        lastState = state;
        if (state != WifiPhyState::SLEEP)
        {
            return;
        }
        const Time a = Max(start, winStart);
        const Time b = Min(end, winEnd);
        if (b > a)
        {
            sleep += b - a;
        }
    }

    /// Account for the state still in progress at the end of the window.
    void Close(Ptr<WifiPhy> phy)
    {
        if (phy->IsStateSleep() && lastEnd < winEnd)
        {
            sleep += winEnd - Max(lastEnd, winStart);
        }
        closed = true;
    }
};

double
Percentile(const Histogram& h, double q)
{
    uint64_t total = 0;
    for (uint32_t i = 0; i < h.GetNBins(); ++i)
    {
        total += h.GetBinCount(i);
    }
    if (total == 0)
    {
        return 0.0;
    }
    const double target = q * total;
    uint64_t acc = 0;
    for (uint32_t i = 0; i < h.GetNBins(); ++i)
    {
        acc += h.GetBinCount(i);
        if (acc >= target)
        {
            return h.GetBinEnd(i);
        }
    }
    return h.GetBinEnd(h.GetNBins() - 1);
}

} // namespace

int
main(int argc, char* argv[])
{
    std::string mode{"twt"};
    std::string traffic{"iot"};
    uint32_t nSta{10};
    double simTime{30.0};
    double warmup{1.0};
    double wakeIntervalMs{100.0};
    double spDurationMs{10.0};
    bool twtStagger{true};
    bool twtEarlySleep{false};
    double radius{10.0};
    double macQueueMaxDelayMs{2000.0};
    uint32_t pktSize{0};
    double pktIntervalMs{0.0};
    std::string outFile{"twt-sim.csv"};
    bool verbose{false};
    int32_t traceSta{-1};

    CommandLine cmd(__FILE__);
    cmd.AddValue("mode", "cam | psm | twt", mode);
    cmd.AddValue("traffic", "iot (100 B / 100 ms) | video (1200 B / 4.8 ms, ~2 Mb/s CBR)", traffic);
    cmd.AddValue("nSta", "number of stations", nSta);
    cmd.AddValue("simTime", "measurement duration (s)", simTime);
    cmd.AddValue("warmup", "time before traffic starts (s)", warmup);
    cmd.AddValue("wakeIntervalMs", "TWT wake interval (ms)", wakeIntervalMs);
    cmd.AddValue("spDurationMs", "TWT service period duration (ms)", spDurationMs);
    cmd.AddValue("twtStagger", "spread STAs' SPs evenly over the wake interval", twtStagger);
    cmd.AddValue("twtEarlySleep", "doze before SP end when nothing left to send", twtEarlySleep);
    cmd.AddValue("radius", "STAs placed uniformly in a disc of this radius (m)", radius);
    cmd.AddValue("macQueueMaxDelayMs", "WifiMacQueue MaxDelay (ms)", macQueueMaxDelayMs);
    cmd.AddValue("pktSize", "override packet size (bytes)", pktSize);
    cmd.AddValue("pktIntervalMs", "override packet interval (ms)", pktIntervalMs);
    cmd.AddValue("outFile", "per-station CSV output", outFile);
    cmd.AddValue("verbose", "print a per-run summary", verbose);
    cmd.AddValue("traceSta", "print PHY state periods of this STA (debug)", traceSta);
    cmd.Parse(argc, argv);

    NS_ABORT_MSG_IF(mode != "cam" && mode != "psm" && mode != "twt", "unknown mode " << mode);
    NS_ABORT_MSG_IF(mode == "twt" && spDurationMs > wakeIntervalMs, "SP longer than interval");

    if (pktSize == 0)
    {
        pktSize = (traffic == "video") ? 1200 : 100;
    }
    if (pktIntervalMs <= 0)
    {
        pktIntervalMs = (traffic == "video") ? 4.8 : 100.0;
    }

    // ---- defaults
    Config::SetDefault("ns3::WifiMacQueue::MaxDelay", TimeValue(MilliSeconds(macQueueMaxDelayMs)));
    // dozing STAs miss beacons by design; don't let the beacon watchdog disassociate them
    Config::SetDefault("ns3::StaWifiMac::MaxMissedBeacons", UintegerValue(1000000));

    // ---- nodes, PHY, MAC
    NodeContainer apNode(1);
    NodeContainer staNodes(nSta);

    WifiHelper wifi;
    wifi.SetStandard(WIFI_STANDARD_80211ax);
    wifi.SetRemoteStationManager("ns3::IdealWifiManager");

    YansWifiChannelHelper channel = YansWifiChannelHelper::Default();
    YansWifiPhyHelper phy;
    phy.SetChannel(channel.Create());
    phy.Set("ChannelSettings", StringValue("{36, 20, BAND_5GHZ, 0}"));

    Ssid ssid("twt-sim");
    WifiMacHelper mac;
    mac.SetType("ns3::ApWifiMac", "Ssid", SsidValue(ssid));
    NetDeviceContainer apDev = wifi.Install(phy, mac, apNode);

    mac.SetType("ns3::StaWifiMac", "Ssid", SsidValue(ssid));
    if (mode == "psm")
    {
        mac.SetPowerSaveManager("ns3::DefaultPowerSaveManager",
                                "PowerSaveMode",
                                StringValue("0 true"));
    }
    else if (mode == "twt")
    {
        mac.SetPowerSaveManager("ns3::TwtPowerSaveManager",
                                "PowerSaveMode",
                                StringValue("0 true"),
                                "WakeInterval",
                                TimeValue(MicroSeconds(std::llround(wakeIntervalMs * 1000))),
                                "SpDuration",
                                TimeValue(MicroSeconds(std::llround(spDurationMs * 1000))),
                                "EarlySleep",
                                BooleanValue(twtEarlySleep));
    }
    NetDeviceContainer staDevs = wifi.Install(phy, mac, staNodes);

    int64_t stream = 100;
    stream += WifiHelper::AssignStreams(apDev, stream);
    stream += WifiHelper::AssignStreams(staDevs, stream);

    // TWT: per-STA SP offsets on a common grid, AP side of the agreement
    auto apMac = DynamicCast<ApWifiMac>(DynamicCast<WifiNetDevice>(apDev.Get(0))->GetMac());
    if (mode == "twt")
    {
        const Time wi = MicroSeconds(std::llround(wakeIntervalMs * 1000));
        for (uint32_t i = 0; i < nSta; ++i)
        {
            auto staMac =
                DynamicCast<StaWifiMac>(DynamicCast<WifiNetDevice>(staDevs.Get(i))->GetMac());
            auto twt = DynamicCast<TwtPowerSaveManager>(staMac->GetPowerSaveManager());
            NS_ASSERT(twt);
            twt->SetApMac(apMac);
            const Time offset = twtStagger ? wi * i / nSta : Time{0};
            twt->SetAttribute("Offset", TimeValue(offset));
        }
    }

    // ---- mobility
    MobilityHelper mobility;
    mobility.SetMobilityModel("ns3::ConstantPositionMobilityModel");
    mobility.Install(apNode);
    mobility.SetPositionAllocator("ns3::UniformDiscPositionAllocator",
                                  "rho",
                                  DoubleValue(radius),
                                  "X",
                                  DoubleValue(0.0),
                                  "Y",
                                  DoubleValue(0.0));
    mobility.Install(staNodes);

    // ---- static association and Block Ack (no association / ADDBA frames to model)
    auto apWifiDev = DynamicCast<WifiNetDevice>(apDev.Get(0));
    WifiStaticSetupHelper::SetStaticAssociation(apWifiDev, staDevs);
    WifiStaticSetupHelper::SetStaticBlockAck(apWifiDev, staDevs, {0});

    // ---- energy (stations only)
    BasicEnergySourceHelper energySource;
    energySource.Set("BasicEnergySourceInitialEnergyJ", DoubleValue(1e9));
    energy::EnergySourceContainer sources = energySource.Install(staNodes);
    WifiRadioEnergyModelHelper radioEnergy; // ns-3 default currents
    energy::DeviceEnergyModelContainer models = radioEnergy.Install(staDevs, sources);

    // ---- IP, static ARP, traffic
    InternetStackHelper internet;
    internet.Install(apNode);
    internet.Install(staNodes);
    Ipv4AddressHelper ip;
    ip.SetBase("10.1.0.0", "255.255.0.0");
    Ipv4InterfaceContainer apIf = ip.Assign(apDev);
    Ipv4InterfaceContainer staIf = ip.Assign(staDevs);
    NeighborCacheHelper().PopulateNeighborCache();

    const uint16_t port = 9;
    PacketSinkHelper sink("ns3::UdpSocketFactory", InetSocketAddress(Ipv4Address::GetAny(), port));
    sink.Install(apNode.Get(0)).Start(Seconds(0));

    const Time tStart = Seconds(warmup);
    const Time tEnd = Seconds(warmup + simTime);
    const Time interval = MicroSeconds(std::llround(pktIntervalMs * 1000));
    auto jitter = CreateObject<UniformRandomVariable>();
    jitter->SetStream(stream++);

    UdpClientHelper client(apIf.GetAddress(0), port);
    client.SetAttribute("MaxPackets", UintegerValue(0));
    client.SetAttribute("Interval", TimeValue(interval));
    client.SetAttribute("PacketSize", UintegerValue(pktSize));
    for (uint32_t i = 0; i < nSta; ++i)
    {
        auto app = client.Install(staNodes.Get(i));
        // random phase so stations don't generate packets in lockstep
        app.Start(tStart + NanoSeconds(jitter->GetInteger(0, interval.GetNanoSeconds() - 1)));
        app.Stop(tEnd);
    }

    FlowMonitorHelper flowHelper;
    flowHelper.SetMonitorAttribute("DelayBinWidth", DoubleValue(0.0001)); // 0.1 ms bins
    Ptr<FlowMonitor> monitor = flowHelper.InstallAll();

    // ---- sleep-time and energy bookkeeping over [tStart, tEnd]
    std::vector<SleepTracker> sleepTrackers(nSta);
    std::vector<double> energyAtStart(nSta, 0.0), energyAtEnd(nSta, 0.0);
    for (uint32_t i = 0; i < nSta; ++i)
    {
        sleepTrackers[i].winStart = tStart;
        sleepTrackers[i].winEnd = tEnd;
        auto staPhy = DynamicCast<WifiNetDevice>(staDevs.Get(i))->GetPhy();
        staPhy->GetState()->TraceConnectWithoutContext(
            "State",
            MakeCallback(&SleepTracker::Add, &sleepTrackers[i]));
    }
    if (traceSta >= 0 && traceSta < static_cast<int32_t>(nSta))
    {
        DynamicCast<WifiNetDevice>(staDevs.Get(traceSta))
            ->GetPhy()
            ->GetState()
            ->TraceConnectWithoutContext(
                "State",
                MakeCallback(+[](Time start, Time duration, WifiPhyState state) {
                    std::cout << "PHY " << start.As(Time::US) << " +" << duration.As(Time::US)
                              << " " << state << "\n";
                }));
    }
    Simulator::Schedule(tStart, [&]() {
        for (uint32_t i = 0; i < nSta; ++i)
        {
            energyAtStart[i] = models.Get(i)->GetTotalEnergyConsumption();
        }
    });
    Simulator::Schedule(tEnd, [&]() {
        for (uint32_t i = 0; i < nSta; ++i)
        {
            energyAtEnd[i] = models.Get(i)->GetTotalEnergyConsumption();
            sleepTrackers[i].Close(DynamicCast<WifiNetDevice>(staDevs.Get(i))->GetPhy());
        }
    });

    // drain: long enough for frames waiting for the next SP (and MAC-queue timeouts)
    Time drain = Seconds(1);
    if (mode == "twt")
    {
        drain = Max(drain, MilliSeconds(2 * wakeIntervalMs));
    }
    drain = Max(drain, MilliSeconds(macQueueMaxDelayMs) + MilliSeconds(100));
    Simulator::Stop(tEnd + drain);
    Simulator::Run();

    // ---- per-station output
    monitor->CheckForLostPackets();
    auto classifier = DynamicCast<Ipv4FlowClassifier>(flowHelper.GetClassifier());
    std::map<Ipv4Address, FlowMonitor::FlowStats> byStaAddr;
    for (const auto& [flowId, st] : monitor->GetFlowStats())
    {
        auto t = classifier->FindFlow(flowId);
        if (t.destinationAddress == apIf.GetAddress(0))
        {
            byStaAddr[t.sourceAddress] = st;
        }
    }

    std::ofstream out(outFile);
    NS_ABORT_MSG_IF(!out, "cannot write " << outFile);
    out << "staId,energyJ,txPackets,rxPackets,rxBytes,meanDelayMs,p95DelayMs,sleepFraction\n";
    out << std::fixed;
    double sumE = 0, sumSleep = 0;
    uint64_t sumTx = 0, sumRx = 0;
    for (uint32_t i = 0; i < nSta; ++i)
    {
        FlowMonitor::FlowStats st;
        if (auto it = byStaAddr.find(staIf.GetAddress(i)); it != byStaAddr.end())
        {
            st = it->second;
        }
        const double e = energyAtEnd[i] - energyAtStart[i];
        const double meanDelay =
            st.rxPackets ? st.delaySum.GetSeconds() * 1e3 / st.rxPackets : 0.0;
        const double p95 = Percentile(st.delayHistogram, 0.95) * 1e3;
        const double sleepFrac = sleepTrackers[i].sleep.GetSeconds() / simTime;
        out << i << "," << std::setprecision(6) << e << "," << st.txPackets << ","
            << st.rxPackets << "," << st.rxBytes << "," << std::setprecision(4) << meanDelay
            << "," << p95 << "," << sleepFrac << "\n";
        sumE += e;
        sumSleep += sleepFrac;
        sumTx += st.txPackets;
        sumRx += st.rxPackets;
    }
    out.close();

    if (verbose)
    {
        std::cout << "mode=" << mode << " traffic=" << traffic << " nSta=" << nSta
                  << " energy/STA=" << sumE / nSta << " J"
                  << " sleep=" << 100 * sumSleep / nSta << "%"
                  << " loss=" << (sumTx ? 100.0 * (sumTx - sumRx) / sumTx : 0) << "%\n";
    }

    Simulator::Destroy();
    return 0;
}
