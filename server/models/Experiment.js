const mongoose = require('mongoose');

const experimentSchema = new mongoose.Schema({
  name: { type: String, required: true },
  trainingDatasets: { type: [String], required: true },
  trainingPairs: { type: Number, required: true },
  metrics: {
    sAlpha: { type: Number, required: true },
    ePhi: { type: Number, required: true },
    fBetaW: { type: Number, required: true },
    mae: { type: Number, required: true }
  },
  diagnostics: {
    boundaryIoU: { type: Number, required: true },
    fpr: { type: Number, required: true },
    bgLeakage: { type: Number, required: true }
  },
  notes: { type: String, required: true }
}, { timestamps: true });

module.exports = mongoose.model('Experiment', experimentSchema);
