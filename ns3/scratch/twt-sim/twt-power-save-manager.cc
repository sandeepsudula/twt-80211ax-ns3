#include "twt-power-save-manager.h"

#include "ns3/boolean.h"
#include "ns3/log.h"
#include "ns3/qos-txop.h"
#include "ns3/simulator.h"
#include "ns3/sta-wifi-mac.h"
#include "ns3/wifi-mac-queue-scheduler.h"
#include "ns3/wifi-phy.h"

namespace ns3
{

NS_LOG_COMPONENT_DEFINE("TwtPowerSaveManager");
NS_OBJECT_ENSURE_REGISTERED(TwtPowerSaveManager);

TypeId
TwtPowerSaveManager::GetTypeId()
{
    static TypeId tid =
        TypeId("ns3::TwtPowerSaveManager")
            .SetParent<PowerSaveManager>()
            .SetGroupName("Wifi")
            .AddConstructor<TwtPowerSaveManager>()
            .AddAttribute("WakeInterval",
                          "TWT wake interval (time between the starts of consecutive SPs).",
                          TimeValue(MilliSeconds(100)),
                          MakeTimeAccessor(&TwtPowerSaveManager::m_wakeInterval),
                          MakeTimeChecker(MicroSeconds(1)))
            .AddAttribute("SpDuration",
                          "TWT service period (minimum wake) duration.",
                          TimeValue(MilliSeconds(10)),
                          MakeTimeAccessor(&TwtPowerSaveManager::m_spDuration),
                          MakeTimeChecker(MicroSeconds(1)))
            .AddAttribute("Offset",
                          "Start of this STA's SPs relative to the common grid "
                          "(SP k starts at Offset + k * WakeInterval).",
                          TimeValue(Time{0}),
                          MakeTimeAccessor(&TwtPowerSaveManager::m_offset),
                          MakeTimeChecker(Time{0}))
            .AddAttribute("EarlySleep",
                          "Doze before the end of the SP once the STA has nothing left to "
                          "send and has no channel access pending.",
                          BooleanValue(false),
                          MakeBooleanAccessor(&TwtPowerSaveManager::m_earlySleep),
                          MakeBooleanChecker())
            .AddAttribute("MinAwake",
                          "With EarlySleep, minimum time to stay awake after the SP starts.",
                          TimeValue(MicroSeconds(500)),
                          MakeTimeAccessor(&TwtPowerSaveManager::m_minAwake),
                          MakeTimeChecker(Time{0}))
            .AddAttribute("PollInterval",
                          "Retry interval for going to sleep while channel access is pending.",
                          TimeValue(MicroSeconds(100)),
                          MakeTimeAccessor(&TwtPowerSaveManager::m_pollInterval),
                          MakeTimeChecker(MicroSeconds(1)));
    return tid;
}

TwtPowerSaveManager::TwtPowerSaveManager()
{
    NS_LOG_FUNCTION(this);
}

TwtPowerSaveManager::~TwtPowerSaveManager()
{
    NS_LOG_FUNCTION_NOARGS();
}

void
TwtPowerSaveManager::DoDispose()
{
    m_startEvent.Cancel();
    m_endEvent.Cancel();
    m_pollEvent.Cancel();
    m_apMac = nullptr;
    PowerSaveManager::DoDispose();
}

void
TwtPowerSaveManager::SetApMac(Ptr<ApWifiMac> apMac)
{
    m_apMac = apMac;
}

uint64_t
TwtPowerSaveManager::GetNServicePeriods() const
{
    return m_nSps;
}

// ---------------------------------------------------------------- schedule

void
TwtPowerSaveManager::Activate()
{
    NS_LOG_FUNCTION(this);
    NS_ABORT_MSG_IF(m_spDuration > m_wakeInterval, "TWT SP longer than wake interval");
    if (m_active)
    {
        return;
    }
    m_active = true;

    const Time now = Simulator::Now();
    const Time offset = m_offset % m_wakeInterval;
    // position of "now" within the current wake interval
    Time sinceStart = (now - offset) % m_wakeInterval;
    if (sinceStart.IsStrictlyNegative())
    {
        sinceStart += m_wakeInterval;
    }
    const Time nextStart = m_wakeInterval - sinceStart;

    if (sinceStart < m_spDuration)
    {
        // we are inside an SP already
        m_inSp = true;
        m_spStart = now - sinceStart;
        ++m_nSps;
        OpenGate();
        m_endEvent = Simulator::Schedule(m_spDuration - sinceStart, &TwtPowerSaveManager::EndSp, this);
    }
    else
    {
        m_inSp = false;
        CloseGate();
        TrySleep();
    }
    m_startEvent = Simulator::Schedule(nextStart, &TwtPowerSaveManager::StartSp, this);
}

void
TwtPowerSaveManager::StartSp()
{
    NS_LOG_FUNCTION(this);
    m_inSp = true;
    m_spStart = Simulator::Now();
    ++m_nSps;
    m_pollEvent.Cancel();

    if (auto phy = GetStaMac()->GetWifiPhy(m_linkId); phy->IsStateSleep())
    {
        NS_LOG_DEBUG("SP start: waking up PHY");
        phy->ResumeFromSleep();
    }
    OpenGate(); // after waking up, so that channel access can be requested

    m_endEvent = Simulator::Schedule(m_spDuration, &TwtPowerSaveManager::EndSp, this);
    m_startEvent = Simulator::Schedule(m_wakeInterval, &TwtPowerSaveManager::StartSp, this);

    if (m_earlySleep)
    {
        m_pollEvent = Simulator::Schedule(m_minAwake, &TwtPowerSaveManager::TrySleep, this);
    }
}

void
TwtPowerSaveManager::EndSp()
{
    NS_LOG_FUNCTION(this);
    m_inSp = false;
    CloseGate();
    TrySleep();
}

// ---------------------------------------------------------------- gating

void
TwtPowerSaveManager::OpenGate()
{
    if (m_gateOpen)
    {
        return;
    }
    m_gateOpen = true;
    auto sta = GetStaMac();
    sta->UnblockUnicastTxOnLinks(WifiQueueBlockedReason::POWER_SAVE_MODE,
                                 sta->GetBssid(m_linkId),
                                 {m_linkId});
    if (m_apMac)
    {
        m_apMac->UnblockUnicastTxOnLinks(WifiQueueBlockedReason::POWER_SAVE_MODE,
                                         sta->GetAddress(),
                                         {m_linkId});
    }
}

void
TwtPowerSaveManager::CloseGate()
{
    if (!m_gateOpen)
    {
        return;
    }
    m_gateOpen = false;
    auto sta = GetStaMac();
    sta->BlockUnicastTxOnLinks(WifiQueueBlockedReason::POWER_SAVE_MODE,
                               sta->GetBssid(m_linkId),
                               {m_linkId});
    if (m_apMac)
    {
        m_apMac->BlockUnicastTxOnLinks(WifiQueueBlockedReason::POWER_SAVE_MODE,
                                       sta->GetAddress(),
                                       {m_linkId});
    }
}

// ---------------------------------------------------------------- sleeping

bool
TwtPowerSaveManager::HasUplinkFrames() const
{
    auto sta = GetStaMac();
    for (const auto ac : {AC_BE, AC_BK, AC_VI, AC_VO})
    {
        if (sta->GetQosTxop(ac)->HasFramesToTransmit(m_linkId))
        {
            return true;
        }
    }
    return false;
}

bool
TwtPowerSaveManager::MayDozeNow() const
{
    if (!m_inSp)
    {
        return true;
    }
    return m_earlySleep && Simulator::Now() - m_spStart >= m_minAwake && !HasUplinkFrames();
}

void
TwtPowerSaveManager::TrySleep()
{
    NS_LOG_FUNCTION(this);
    m_pollEvent.Cancel();

    if (!m_active || !MayDozeNow())
    {
        return;
    }
    auto sta = GetStaMac();
    if (sta->GetPmMode(m_linkId) != WIFI_PM_POWERSAVE)
    {
        return;
    }
    auto phy = sta->GetWifiPhy(m_linkId);
    if (phy->IsStateSleep())
    {
        return;
    }
    if (HasRequestedOrGainedChannel(m_linkId) || phy->IsStateTx() || phy->IsStateRx())
    {
        // finish the ongoing exchange (or let a blocked request be released) first
        m_pollEvent = Simulator::Schedule(m_pollInterval, &TwtPowerSaveManager::TrySleep, this);
        return;
    }
    NS_LOG_DEBUG("PHY to sleep (inSp=" << m_inSp << ")");
    phy->SetSleepMode(true);
}

// ---------------------------------------------------------------- notifications

void
TwtPowerSaveManager::DoNotifyAssocCompleted()
{
    NS_LOG_FUNCTION(this);
    if (GetStaMac()->GetPmMode(m_linkId) == WIFI_PM_POWERSAVE)
    {
        Activate();
    }
    // otherwise wait for the switch to PS mode to be acknowledged
}

void
TwtPowerSaveManager::DoNotifyDisassociation()
{
    NS_LOG_FUNCTION(this);
    m_active = false;
    m_inSp = false;
    m_startEvent.Cancel();
    m_endEvent.Cancel();
    m_pollEvent.Cancel();
}

void
TwtPowerSaveManager::DoNotifyPmModeChanged(WifiPowerManagementMode pmMode, linkId_t linkId)
{
    NS_LOG_FUNCTION(this << pmMode << linkId);
    if (pmMode == WIFI_PM_POWERSAVE && GetStaMac()->IsAssociated())
    {
        Activate();
    }
}

void
TwtPowerSaveManager::DoNotifyReceivedBeacon(const MgtBeaconHeader&, linkId_t)
{
    // TWT STAs do not need Beacons to schedule wake-ups; nothing to do
}

void
TwtPowerSaveManager::DoNotifyReceivedFrameAfterPsPoll(Ptr<const WifiMpdu>, linkId_t)
{
}

void
TwtPowerSaveManager::DoNotifyReceivedGroupcast(Ptr<const WifiMpdu>, linkId_t)
{
}

void
TwtPowerSaveManager::DoNotifyRequestAccess(Ptr<Txop> txop, linkId_t linkId)
{
    NS_LOG_FUNCTION(this << txop << linkId);
    // Outside an SP the request is deferred: the PHY is NOT woken up, and the frame
    // contends at the next SP start. Inside an SP (early-sleep case) wake up now.
    if (m_inSp)
    {
        m_pollEvent.Cancel();
        if (auto phy = GetStaMac()->GetWifiPhy(linkId); phy->IsStateSleep())
        {
            phy->ResumeFromSleep();
        }
    }
}

void
TwtPowerSaveManager::DoNotifyChannelReleased(Ptr<Txop> txop, linkId_t linkId)
{
    NS_LOG_FUNCTION(this << txop << linkId);
    if (!m_inSp || m_earlySleep)
    {
        // schedule now: the Txop may re-request access right after releasing
        Simulator::ScheduleNow(&TwtPowerSaveManager::TrySleep, this);
    }
}

void
TwtPowerSaveManager::DoTxDropped(WifiMacDropReason, Ptr<const WifiMpdu>)
{
    if (!m_inSp || m_earlySleep)
    {
        Simulator::ScheduleNow(&TwtPowerSaveManager::TrySleep, this);
    }
}

} // namespace ns3
