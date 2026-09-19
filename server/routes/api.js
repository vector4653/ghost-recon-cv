const express = require('express');
const router = express.Router();
const multer = require('multer');
const axios = require('axios');
const fs = require('fs');
const path = require('path');
const FormData = require('form-data');
const Experiment = require('../models/Experiment');
const InferenceLog = require('../models/InferenceLog');

const upload = multer({ dest: 'uploads/' });

// Fallback data in case DB is not populated
const fallbackExperiments = [
  { name: 'Baseline', sAlpha: 0.7647, ePhi: 0.8061, fBetaW: 0.6036, mae: 0.0593, boundaryIoU: 0.2332, fpr: 0.0141 },
  { name: 'Exp M', sAlpha: 0.8638, ePhi: 0.9046, fBetaW: 0.7718, mae: 0.0348, boundaryIoU: 0.3372, fpr: 0.0202 },
  { name: 'Exp C1', sAlpha: 0.7572, ePhi: 0.8045, fBetaW: 0.5886, mae: 0.0538, boundaryIoU: 0.2225, fpr: 0.0158 },
  { name: 'Exp C2', sAlpha: 0.6223, ePhi: 0.6616, fBetaW: 0.3541, mae: 0.1007, boundaryIoU: 0.1414, fpr: 0.0081 },
  { name: 'Exp MC1 (Winner)', sAlpha: 0.8675, ePhi: 0.9095, fBetaW: 0.7773, mae: 0.0322, boundaryIoU: 0.3437, fpr: 0.0155 },
  { name: 'Exp MC2', sAlpha: 0.8460, ePhi: 0.8765, fBetaW: 0.7276, mae: 0.0423, boundaryIoU: 0.3119, fpr: 0.0219 },
  { name: 'Exp MCC', sAlpha: 0.8518, ePhi: 0.8802, fBetaW: 0.7379, mae: 0.0408, boundaryIoU: 0.3199, fpr: 0.0199 }
];

// Preset tactical scenarios pulled from the Military Test split, when mounted locally
const scenarioLabels = ['Concealed Sniper', 'Woodland Foliage', 'Prone Stance', 'Urban Ambush', 'Treeline Watch', 'Ridge Overwatch'];
const IMAGE_EXT = new Set(['.jpg', '.jpeg', '.png', '.bmp']);

router.get('/presets', (req, res) => {
  const imagesDir = path.join(__dirname, '../../Dataset/TestDataset/Images');
  if (!fs.existsSync(imagesDir)) {
    return res.json({ mounted: false, presets: [] });
  }

  try {
    const files = fs.readdirSync(imagesDir).filter((f) => IMAGE_EXT.has(path.extname(f).toLowerCase()));
    const presets = files.slice(0, scenarioLabels.length).map((file, i) => ({
      label: scenarioLabels[i] || file,
      filename: file,
      url: `/dataset/${file}`,
    }));
    res.json({ mounted: true, presets });
  } catch (error) {
    res.json({ mounted: false, presets: [] });
  }
});

router.get('/experiments', async (req, res) => {
  try {
    const experiments = await Experiment.find();
    if (experiments.length > 0) {
      res.json(experiments);
    } else {
      res.json(fallbackExperiments);
    }
  } catch (error) {
    res.json(fallbackExperiments);
  }
});

router.post('/infer', upload.single('image'), async (req, res) => {
  if (!req.file) {
    return res.status(400).json({ error: 'No image provided' });
  }

  const modelSelected = req.body.model || 'Exp MC1';
  
  try {
    const formData = new FormData();
    formData.append('image', fs.createReadStream(req.file.path));
    formData.append('model', modelSelected);

    // Call FastAPI Sidecar
    const fastApiUrl = process.env.FASTAPI_URL || 'http://localhost:8000';
    const startTime = Date.now();
    const response = await axios.post(`${fastApiUrl}/predict`, formData, {
      headers: formData.getHeaders()
    });
    const executionTimeMs = Date.now() - startTime;

    const inferenceData = {
      filename: req.file.originalname,
      modelSelected,
      executionTimeMs,
      telemetry: response.data.telemetry
    };

    try {
      await InferenceLog.create(inferenceData);
    } catch (dbError) {
      console.warn('Could not save to MongoDB, continuing...');
    }

    res.json({
      maskBase64: response.data.maskBase64,
      telemetry: response.data.telemetry,
      executionTimeMs
    });
  } catch (error) {
    console.error('Inference error:', error.message);
    res.status(500).json({ error: 'Failed to process inference' });
  } finally {
    fs.unlink(req.file.path, () => {});
  }
});

router.get('/history', async (req, res) => {
  try {
    const history = await InferenceLog.find().sort({ timestamp: -1 }).limit(10);
    res.json(history);
  } catch (error) {
    res.json([]);
  }
});

router.get('/paper', (req, res) => {
  const paperPath = path.join(__dirname, '../../paper/RESEARCH_PAPER.md');
  if (fs.existsSync(paperPath)) {
    const content = fs.readFileSync(paperPath, 'utf-8');
    res.json({ content });
  } else {
    res.status(404).json({ error: 'Paper not found' });
  }
});

module.exports = router;
