export default (io) => {
  io.on("connection", (socket) => {
    console.log("🌐 Frontend connected to WebSocket");
  });
};
