const mongoose = require('mongoose');

const inferenceLogSchema = new mongoose.Schema({
  filename: { type: String, required: true },
  modelSelected: { type: String, required: true },
  executionTimeMs: { type: Number, required: true },
  telemetry: {
    camouflageAreaPercent: { type: Number, required: true },
    confidenceScore: { type: Number, required: true },
    threatLevel: { type: String, required: true }
  },
  timestamp: { type: Date, default: Date.now }
});

module.exports = mongoose.model('InferenceLog', inferenceLogSchema);
