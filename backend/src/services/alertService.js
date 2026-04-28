import nodemailer from "nodemailer";
import twilio from "twilio";

export const sendEmailAlert = async (subject, message) => {
  if (!process.env.EMAIL || !process.env.EMAIL_PASS || !process.env.ALERT_EMAIL) {
    console.warn("Email alert skipped — email credentials or recipient not configured.");
    return;
  }

  try {
    const transporter = nodemailer.createTransport({
      service: "gmail",
      auth: {
        user: process.env.EMAIL,
        pass: process.env.EMAIL_PASS,
      },
    });

    await transporter.sendMail({
      from: process.env.EMAIL,
      to: process.env.ALERT_EMAIL,
      subject,
      text: message,
    });

    console.log("📧 Email Alert Sent");
  } catch (err) {
    console.error("Failed to send email alert:", err.message || err);
  }
};

export const sendSMSAlert = async (message) => {
  if (!process.env.TWILIO_SID || !process.env.TWILIO_AUTH || !process.env.TWILIO_PHONE || !process.env.ALERT_PHONE) {
    console.warn("SMS alert skipped — Twilio credentials or phone numbers not configured.");
    return;
  }

  try {
    const client = twilio(process.env.TWILIO_SID, process.env.TWILIO_AUTH);

    await client.messages.create({
      body: message,
      from: process.env.TWILIO_PHONE,
      to: process.env.ALERT_PHONE,
    });

    console.log("📱 SMS Alert Sent");
  } catch (err) {
    console.error("Failed to send SMS alert:", err.message || err);
  }
};
