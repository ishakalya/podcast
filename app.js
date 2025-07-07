const express = require('express');
const mongoose = require('mongoose');
const cors = require('cors');
const bcrypt = require('bcryptjs');
const jwt = require('jsonwebtoken');
require('dotenv').config();
const path = require('path');

const app = express();
const PORT = process.env.PORT || 5050;

// Basic middleware
app.use(cors());
app.use(express.json({ limit: '10mb' }));
app.use(express.static('public'));

// MongoDB connection with better error handling
const connectDB = async () => {
  try {
    await mongoose.connect(process.env.MONGODB_URI || 'mongodb://localhost:27017/podcast-streaming', {
      serverSelectionTimeoutMS: 5000,
      maxPoolSize: 10,
      bufferMaxEntries: 0,
      bufferCommands: false,
      connectTimeoutMS: 30000,
      socketTimeoutMS: 30000
    });
    console.log('✅ Connected to MongoDB successfully');
  } catch (err) {
    console.error('❌ MongoDB connection failed:', err);
    console.log('📄 Running in demo mode without database');
    // Set mongoose to not buffer commands when disconnected
    mongoose.set('bufferCommands', false);
  }
};

// User Schema
const userSchema = new mongoose.Schema({
  username: { type: String, required: true, unique: true },
  email: { type: String, required: true, unique: true },
  password: { type: String, required: true },
  firstName: { type: String, required: true },
  lastName: { type: String, required: true },
  role: { type: String, enum: ['listener', 'creator', 'admin'], default: 'listener' }
}, { timestamps: true });

userSchema.pre('save', async function(next) {
  if (!this.isModified('password')) return next();
  this.password = await bcrypt.hash(this.password, 12);
  next();
});

userSchema.methods.comparePassword = async function(password) {
  return await bcrypt.compare(password, this.password);
};

const User = mongoose.model('User', userSchema);

// Podcast Schema
const podcastSchema = new mongoose.Schema({
  title: { type: String, required: true },
  description: { type: String, required: true },
  creator: { type: mongoose.Schema.Types.ObjectId, ref: 'User', required: true },
  category: { type: String, required: true },
  coverImage: String,
  episodes: [{
    title: { type: String, required: true },
    description: { type: String, required: true },
    audioFile: String,
    duration: Number,
    episodeNumber: Number,
    publishDate: { type: Date, default: Date.now }
  }],
  subscribers: [{ type: mongoose.Schema.Types.ObjectId, ref: 'User' }],
  isActive: { type: Boolean, default: true }
}, { timestamps: true });

const Podcast = mongoose.model('Podcast', podcastSchema);

// Auth middleware
const auth = async (req, res, next) => {
  try {
    const token = req.header('Authorization')?.replace('Bearer ', '');
    if (!token) {
      return res.status(401).json({ success: false, message: 'Access denied' });
    }

    const decoded = jwt.verify(token, process.env.JWT_SECRET || 'fallback-secret');
    const user = await User.findById(decoded.userId);
    if (!user) {
      return res.status(401).json({ success: false, message: 'Invalid token' });
    }

    req.user = user;
    next();
  } catch (error) {
    res.status(401).json({ success: false, message: 'Invalid token' });
  }
};

// Health check
app.get('/', (req, res) => {
  res.json({ 
    message: '🎧 Podcast Streaming API', 
    status: 'running',
    timestamp: new Date().toISOString()
  });
});

app.get('/api/health', (req, res) => {
  res.json({ 
    status: 'OK', 
    message: 'API is healthy',
    timestamp: new Date().toISOString()
  });
});

// Register
app.post('/api/auth/register', async (req, res) => {
  try {
    const { username, email, password, firstName, lastName, role } = req.body;

    const existingUser = await User.findOne({ 
      $or: [{ email }, { username }] 
    });

    if (existingUser) {
      return res.status(400).json({
        success: false,
        message: 'User already exists with this email or username'
      });
    }

    const user = new User({
      username,
      email,
      password,
      firstName,
      lastName,
      role: role || 'listener'
    });

    await user.save();

    const token = jwt.sign(
      { userId: user._id },
      process.env.JWT_SECRET || 'fallback-secret',
      { expiresIn: '7d' }
    );

    res.status(201).json({
      success: true,
      message: 'User registered successfully',
      token,
      user: {
        id: user._id,
        username: user.username,
        email: user.email,
        firstName: user.firstName,
        lastName: user.lastName,
        role: user.role
      }
    });
  } catch (error) {
    console.error('Registration error:', error);
    res.status(500).json({
      success: false,
      message: 'Registration failed'
    });
  }
});

// Login
app.post('/api/auth/login', async (req, res) => {
  try {
    const { email, password } = req.body;

    const user = await User.findOne({ email });
    if (!user || !await user.comparePassword(password)) {
      return res.status(401).json({
        success: false,
        message: 'Invalid credentials'
      });
    }

    const token = jwt.sign(
      { userId: user._id },
      process.env.JWT_SECRET || 'fallback-secret',
      { expiresIn: '7d' }
    );

    res.json({
      success: true,
      message: 'Login successful',
      token,
      user: {
        id: user._id,
        username: user.username,
        email: user.email,
        firstName: user.firstName,
        lastName: user.lastName,
        role: user.role
      }
    });
  } catch (error) {
    console.error('Login error:', error);
    res.status(500).json({
      success: false,
      message: 'Login failed'
    });
  }
});

// Get profile
app.get('/api/auth/profile', auth, async (req, res) => {
  res.json({
    success: true,
    user: {
      id: req.user._id,
      username: req.user.username,
      email: req.user.email,
      firstName: req.user.firstName,
      lastName: req.user.lastName,
      role: req.user.role
    }
  });
});

// Get categories
app.get('/api/podcasts/categories', (req, res) => {
  res.json({
    success: true,
    categories: [
      'Technology', 'Business', 'Education', 'Health & Fitness',
      'News & Politics', 'Entertainment', 'Sports', 'Music',
      'Comedy', 'History', 'Science', 'Arts'
    ]
  });
});

// Get all podcasts
app.get('/api/podcasts', async (req, res) => {
  try {
    const { category, search, page = 1, limit = 12 } = req.query;
    
    let filter = { isActive: true };
    
    if (category) {
      filter.category = category;
    }
    
    if (search) {
      filter.$or = [
        { title: { $regex: search, $options: 'i' } },
        { description: { $regex: search, $options: 'i' } }
      ];
    }

    const podcasts = await Podcast.find(filter)
      .populate('creator', 'username firstName lastName')
      .sort({ createdAt: -1 })
      .limit(parseInt(limit))
      .skip((parseInt(page) - 1) * parseInt(limit));

    const total = await Podcast.countDocuments(filter);

    res.json({
      success: true,
      podcasts: podcasts.map(p => ({
        ...p.toObject(),
        episodeCount: p.episodes.length,
        subscriberCount: p.subscribers.length
      })),
      pagination: {
        currentPage: parseInt(page),
        totalPages: Math.ceil(total / parseInt(limit)),
        totalPodcasts: total
      }
    });
  } catch (error) {
    console.error('Get podcasts error:', error);
    res.status(500).json({
      success: false,
      message: 'Failed to fetch podcasts'
    });
  }
});

// Get specific podcast
app.get('/api/podcasts/:id', async (req, res) => {
  try {
    const podcast = await Podcast.findOne({ 
      _id: req.params.id, 
      isActive: true 
    }).populate('creator', 'username firstName lastName');

    if (!podcast) {
      return res.status(404).json({
        success: false,
        message: 'Podcast not found'
      });
    }

    res.json({
      success: true,
      podcast: {
        ...podcast.toObject(),
        episodeCount: podcast.episodes.length,
        subscriberCount: podcast.subscribers.length
      }
    });
  } catch (error) {
    console.error('Get podcast error:', error);
    res.status(500).json({
      success: false,
      message: 'Failed to fetch podcast'
    });
  }
});

// Create podcast
app.post('/api/podcasts', auth, async (req, res) => {
  try {
    if (req.user.role !== 'creator' && req.user.role !== 'admin') {
      return res.status(403).json({
        success: false,
        message: 'Creator privileges required'
      });
    }

    const { title, description, category } = req.body;

    const podcast = new Podcast({
      title,
      description,
      category,
      creator: req.user._id
    });

    await podcast.save();
    await podcast.populate('creator', 'username firstName lastName');

    res.status(201).json({
      success: true,
      message: 'Podcast created successfully',
      podcast
    });
  } catch (error) {
    console.error('Create podcast error:', error);
    res.status(500).json({
      success: false,
      message: 'Failed to create podcast'
    });
  }
});

// Subscribe to podcast
app.post('/api/podcasts/:id/subscribe', auth, async (req, res) => {
  try {
    const podcast = await Podcast.findById(req.params.id);
    
    if (!podcast) {
      return res.status(404).json({
        success: false,
        message: 'Podcast not found'
      });
    }

    if (!podcast.subscribers.includes(req.user._id)) {
      podcast.subscribers.push(req.user._id);
      await podcast.save();
    }

    res.json({
      success: true,
      message: 'Successfully subscribed'
    });
  } catch (error) {
    console.error('Subscribe error:', error);
    res.status(500).json({
      success: false,
      message: 'Failed to subscribe'
    });
  }
});

// Get user subscriptions
app.get('/api/users/subscriptions', auth, async (req, res) => {
  try {
    const podcasts = await Podcast.find({
      subscribers: req.user._id,
      isActive: true
    }).populate('creator', 'username firstName lastName');

    res.json({
      success: true,
      subscriptions: podcasts.map(p => ({
        ...p.toObject(),
        episodeCount: p.episodes.length,
        subscriberCount: p.subscribers.length
      }))
    });
  } catch (error) {
    console.error('Get subscriptions error:', error);
    res.status(500).json({
      success: false,
      message: 'Failed to fetch subscriptions'
    });
  }
});

// Update podcast
app.put('/api/podcasts/:id', auth, async (req, res) => {
  try {
    const podcast = await Podcast.findById(req.params.id);
    if (!podcast) {
      return res.status(404).json({ success: false, message: 'Podcast not found' });
    }
    // Only creator or admin can update
    if (podcast.creator.toString() !== req.user._id.toString() && req.user.role !== 'admin') {
      return res.status(403).json({ success: false, message: 'Not authorized to update this podcast' });
    }
    const { title, description, category, coverImage, isActive } = req.body;
    if (title !== undefined) podcast.title = title;
    if (description !== undefined) podcast.description = description;
    if (category !== undefined) podcast.category = category;
    if (coverImage !== undefined) podcast.coverImage = coverImage;
    if (isActive !== undefined) podcast.isActive = isActive;
    await podcast.save();
    res.json({ success: true, message: 'Podcast updated successfully', podcast });
  } catch (error) {
    console.error('Update podcast error:', error);
    res.status(500).json({ success: false, message: 'Failed to update podcast' });
  }
});

// Delete podcast
app.delete('/api/podcasts/:id', auth, async (req, res) => {
  try {
    const podcast = await Podcast.findById(req.params.id);
    if (!podcast) {
      return res.status(404).json({ success: false, message: 'Podcast not found' });
    }
    // Only creator or admin can delete
    if (podcast.creator.toString() !== req.user._id.toString() && req.user.role !== 'admin') {
      return res.status(403).json({ success: false, message: 'Not authorized to delete this podcast' });
    }
    await podcast.deleteOne();
    res.json({ success: true, message: 'Podcast deleted successfully' });
  } catch (error) {
    console.error('Delete podcast error:', error);
    res.status(500).json({ success: false, message: 'Failed to delete podcast' });
  }
});

// Error handling
app.use((err, req, res, next) => {
  console.error(err.stack);
  res.status(500).json({
    success: false,
    message: 'Something went wrong!'
  });
});

// 404 handler for API routes only
app.use('/api', (req, res) => {
  res.status(404).json({
    success: false,
    message: 'API route not found'
  });
});

// For all other routes, serve the frontend
app.use((req, res) => {
  res.sendFile(path.join(__dirname, 'public', 'index.html'));
});

// Start server
const startServer = async () => {
  await connectDB();
  app.listen(PORT, '0.0.0.0', () => {
    console.log(`🎧 Podcast Streaming Server running on port ${PORT}`);
    console.log(`🌐 Server URL: http://localhost:${PORT}`);
  });
};

startServer().catch(console.error);

module.exports = app;