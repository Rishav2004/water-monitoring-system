import WaterStatus from "../models/WaterStatus.js";

export const getHistory = async (req, res) => {
  const data = await WaterStatus.find().sort({ createdAt: -1 });
  res.json(data);
};
