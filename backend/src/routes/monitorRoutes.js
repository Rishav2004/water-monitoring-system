import express from "express";
import upload from "../utils/upload.js";
import { analyzeFrame } from "../controllers/monitorController.js";

const router = express.Router();

router.post("/frame", upload.single("image"), analyzeFrame);

export default router;
