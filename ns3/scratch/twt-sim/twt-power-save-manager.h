/*
 * Individual TWT power save manager for ns-3.48 (single-link non-AP STA).
 *
 * The STA stays in Power Management (PM) = power-save mode for the whole
 * simulation, so the AP buffers downlink frames for it. On top of that, the STA
 * and the AP share a fixed individual TWT schedule anchored at absolute
 * simulation time:
 *
 *   SP k starts at  offset + k * wakeInterval  and lasts  spDuration.
 *
 * Inside an SP the STA PHY is awake and unicast frames are allowed in both
 * directions. At the end of an SP:
 *   - the STA's uplink queues towards the AP are frozen (not dropped), so
 *     pending frames and retransmissions wait for the next SP instead of keeping
 *     the radio awake;
 *   - the AP's downlink queues towards the STA are frozen;
 *   - the PHY is put to sleep as soon as no channel access is pending.
 *
 * Gating reuses ns-3's POWER_SAVE_MODE queue-block reason on both sides.
 *
 * The TWT agreement is set up statically (no TWT Setup frames are exchanged),
 * which is the usual assumption when studying TWT parameters rather than the
 * negotiation itself.
 */

#ifndef TWT_POWER_SAVE_MANAGER_H
#define TWT_POWER_SAVE_MANAGER_H

#include "ns3/ap-wifi-mac.h"
#include "ns3/event-id.h"
#include "ns3/nstime.h"
#include "ns3/power-save-manager.h"

namespace ns3
{

class TwtPowerSaveManager : public PowerSaveManager
{
  public:
    static TypeId GetTypeId();
    TwtPowerSaveManager();
    ~TwtPowerSaveManager() override;

    /**
     * Give the manager the AP's MAC so it can gate the AP's downlink queues for
     * this STA at SP boundaries (the AP side of the TWT agreement). Optional:
     * without it, only the STA side is enforced.
     */
    void SetApMac(Ptr<ApWifiMac> apMac);

    /// @return the number of service periods started so far
    uint64_t GetNServicePeriods() const;

  protected:
    void DoDispose() override;

    void DoNotifyAssocCompleted() override;
    void DoNotifyDisassociation() override;
    void DoNotifyPmModeChanged(WifiPowerManagementMode pmMode, linkId_t linkId) override;
    void DoNotifyReceivedBeacon(const MgtBeaconHeader& beacon, linkId_t linkId) override;
    void DoNotifyReceivedFrameAfterPsPoll(Ptr<const WifiMpdu> mpdu, linkId_t linkId) override;
    void DoNotifyReceivedGroupcast(Ptr<const WifiMpdu> mpdu, linkId_t linkId) override;
    void DoNotifyRequestAccess(Ptr<Txop> txop, linkId_t linkId) override;
    void DoNotifyChannelReleased(Ptr<Txop> txop, linkId_t linkId) override;
    void DoTxDropped(WifiMacDropReason reason, Ptr<const WifiMpdu> mpdu) override;

  private:
    void Activate();   ///< start following the TWT schedule
    void StartSp();    ///< beginning of a service period
    void EndSp();      ///< end of a service period
    void TrySleep();   ///< put the PHY to sleep if allowed (polls while channel access is pending)
    void OpenGate();   ///< allow unicast frames between STA and AP
    void CloseGate();  ///< freeze unicast frames between STA and AP
    bool HasUplinkFrames() const;
    bool MayDozeNow() const;

    Time m_wakeInterval; ///< TWT wake interval
    Time m_spDuration;   ///< (minimum) TWT wake duration
    Time m_offset;       ///< offset of this STA's SPs within the wake interval
    bool m_earlySleep;   ///< doze before SP end once nothing is left to send
    Time m_minAwake;     ///< with early sleep: minimum time awake after SP start
    Time m_pollInterval; ///< how often to retry sleeping while channel access is pending

    linkId_t m_linkId{0};
    bool m_active{false};
    bool m_inSp{false};
    bool m_gateOpen{true};
    Time m_spStart;
    uint64_t m_nSps{0};
    Ptr<ApWifiMac> m_apMac;
    EventId m_startEvent;
    EventId m_endEvent;
    EventId m_pollEvent;
};

} // namespace ns3

#endif /* TWT_POWER_SAVE_MANAGER_H */
