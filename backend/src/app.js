import express from "express";
import cors from "cors";
import dotenv from "dotenv";
import { Server } from "socket.io";
import http from "http";

import connectDB from "./config/db.js";
import monitorRoutes from "./routes/monitorRoutes.js";
import historyRoutes from "./routes/historyRoutes.js";
import wsHandler from "./ws.js";

dotenv.config();
connectDB();


const app = express();
app.use(cors());
app.use(express.json());
app.use("/uploads", express.static("uploads"));
app.get("/", (req, res) => res.send("🚀 Water Monitoring Backend is Running"));

const server = http.createServer(app);
const io = new Server(server, { cors: { origin: "*" } });

wsHandler(io);

app.use((req, res, next) => {
  req.io = io;
  next();
});

app.use("/api/monitor", monitorRoutes);
app.use("/api/history", historyRoutes);

server.listen(5000, () =>
  console.log("🚀 Backend running on http://localhost:5000")
);
