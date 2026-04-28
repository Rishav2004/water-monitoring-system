import WaterStatus from "../models/WaterStatus.js";
import { sendEmailAlert, sendSMSAlert } from "../services/alertService.js";

export const checkForAlert = async () => {
  const latest = await WaterStatus.find().sort({ createdAt: -1 }).limit(3);

  if (latest.length < 3) return;

  const pollutedCount = latest.filter(i => i.status === "polluted_water").length;
  const algaeCount = latest.filter(i => i.status === "algae_present").length;

  if (pollutedCount === 3 || algaeCount === 3) {
    const recent = latest[0];

    if (!recent.alertSent) {
      const msg = `⚠ WATER QUALITY ALERT ⚠
Status: ${recent.status}
Time: ${recent.createdAt}
Immediate Action Required!`;

      await sendEmailAlert("Water Quality Alert 🚨", msg);
      await sendSMSAlert(msg);

      recent.alertSent = true;
      await recent.save();
    }
  }
};
