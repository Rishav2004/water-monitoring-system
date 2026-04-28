import mongoose from "mongoose";

const MONGO_TIMEOUT_MS = 5000; // 5 seconds to attempt connection

const connectDB = async () => {
  if (!process.env.MONGO_URI) {
    console.warn("⚠  MONGO_URI is not set in environment.");
    console.warn("⚠  Falling back to local JSON storage (data.json).");
    return;
  }

  try {
    await mongoose.connect(process.env.MONGO_URI, {
      serverSelectionTimeoutMS: MONGO_TIMEOUT_MS,
      connectTimeoutMS: MONGO_TIMEOUT_MS,
    });
    console.log("✔  MongoDB connected successfully.");
  } catch (error) {
    // Surface a clear, actionable message instead of a raw stack trace
    console.warn("⚠  Could not connect to MongoDB:", error.message);
    console.warn("⚠  Falling back to local JSON storage (data.json).");
    console.warn("   → To use MongoDB, make sure it is running and MONGO_URI is correct.");
    // Do NOT exit — the app will continue using the JSON file fallback
  }
};

export default connectDB;
