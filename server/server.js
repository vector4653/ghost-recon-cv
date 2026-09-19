const express = require('express');
const cors = require('cors');
const mongoose = require('mongoose');
const dotenv = require('dotenv');
const morgan = require('morgan');
const path = require('path');
const apiRoutes = require('./routes/api');

dotenv.config();

const app = express();
const PORT = process.env.PORT || 5000;

app.use(cors());
app.use(express.json());
app.use(morgan('dev'));

// Static serving for qualitative analysis images
app.use('/artifacts', express.static(path.join(__dirname, '../qualitative_analysis')));
app.use('/dataset', express.static(path.join(__dirname, '../Dataset/TestDataset/Images')));

app.use('/api', apiRoutes);

const connectDB = async () => {
  try {
    const mongoUri = process.env.MONGO_URI || 'mongodb://localhost:27017/ghost-recon-cv';
    await mongoose.connect(mongoUri);
    console.log('MongoDB Connected');
  } catch (error) {
    console.error('MongoDB connection failed. Running in fallback mode without persistence.');
  }
};

// Start server first, then attempt DB connection in background
app.listen(PORT, () => {
  console.log(`Server running on port ${PORT}`);
});

connectDB();

