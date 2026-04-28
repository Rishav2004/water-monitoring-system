import fs from "fs";
import path from "path";
import mongoose from "mongoose";

const DATA_FILE = path.join(process.cwd(), "data.json");

const loadData = () => {
  if (fs.existsSync(DATA_FILE)) {
    try {
      return JSON.parse(fs.readFileSync(DATA_FILE, "utf-8"));
    } catch (e) {
      console.error("Error reading data.json:", e);
      return [];
    }
  }
  return [];
};

const saveData = (data) => {
  fs.writeFileSync(DATA_FILE, JSON.stringify(data, null, 2));
};

const WaterStatusMock = {
  create: async (data) => {
    const records = loadData();
    const newRecord = {
      ...data,
      latitude: data.latitude || null,
      longitude: data.longitude || null,
      locationName: data.locationName || null,
      _id: Date.now().toString(),
      createdAt: new Date(),
      toObject: function() { return this; }
    };
    records.push(newRecord);
    saveData(records);
    return newRecord;
  },
  find: (query) => {
    const records = loadData();
    // Simplified mock for find().sort().limit()
    const chain = {
      sort: () => chain,
      limit: () => chain,
      exec: async () => records.slice().sort((a, b) => new Date(b.createdAt) - new Date(a.createdAt))
    };
    // If called without exec, it should behave like a promise or have then/catch
    chain.then = (cb) => chain.exec().then(cb);
    return chain;
  }
};

// Check if we should use the real model or mock
let exportModel;
try {
    const waterStatusSchema = new mongoose.Schema({
      status: { type: String, required: true },
      imageUrl: { type: String },
      confidence: { type: Number, default: null },
      latitude: { type: Number, default: null },
      longitude: { type: Number, default: null },
      locationName: { type: String, default: null },
      alertSent: { type: Boolean, default: false },
      createdAt: { type: Date, default: Date.now }
    });
    
    const RealModel = mongoose.models.WaterStatus || mongoose.model("WaterStatus", waterStatusSchema,"waterstatus");
    
    // We will wrap the model to fall back to mock if not connected
    exportModel = {
        create: async (data) => {
            if (mongoose.connection.readyState === 1) {
                return await RealModel.create(data);
            }
            console.warn("[WARNING] MongoDB not connected. Using MOCK storage for create.");
            return await WaterStatusMock.create(data);
        },
        find: (query) => {
            if (mongoose.connection.readyState === 1) {
                return RealModel.find(query);
            }
            console.warn("[WARNING] MongoDB not connected. Using MOCK storage for find.");
            return WaterStatusMock.find(query);
        }
    };
} catch (e) {
    console.error("[ERROR] Failed to initialize Mongoose:", e);
    exportModel = WaterStatusMock;
}

export default exportModel;
