import axios from "axios";
import FormData from "form-data";
import fs from "fs";
import WaterStatus from "../models/WaterStatus.js";
import { checkForAlert } from "../utils/alertChecker.js";

export const analyzeFrame = async (req, res) => {
  try {
    const img = req.file;

    let form = new FormData();
    form.append("image", fs.createReadStream(img.path));

    const flaskRes = await axios.post("http://127.0.0.1:5001/predict", form, {
      headers: form.getHeaders(),
    });

    const status = flaskRes.data.status;
    const confidence = typeof flaskRes.data.confidence === "number" ? flaskRes.data.confidence : null;

    const { latitude, longitude, locationName } = req.body;
    
    const record = await WaterStatus.create({
      status,
      confidence,
      imageUrl: img.filename,
      latitude: latitude ? parseFloat(latitude) : null,
      longitude: longitude ? parseFloat(longitude) : null,
      locationName: locationName || null,
    });

    await checkForAlert();

    // Emit a compact payload for live updates
    req.io.emit("live_update", {
      status,
      confidence,
      imageUrl: img.filename,
      time: record.createdAt.toISOString(),
      latitude: record.latitude,
      longitude: record.longitude,
      locationName: record.locationName,
    });

    // Return the created record with ISO date to avoid frontend `Invalid Date`
    const out = {
      ...record.toObject(),
      createdAt: record.createdAt.toISOString(),
    };

    res.json({
      success: true,
      status,
      record: out,
    });

  } catch (error) {
    console.error(error);
    res.status(500).json({ error: "Prediction failed" });
  }
};
