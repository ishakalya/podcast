const express = require('express');
const mongoose = require('mongoose');
const cors = require('cors');
const bcrypt = require('bcryptjs');
const jwt = require('jsonwebtoken');
require('dotenv').config();

const app = express();
const PORT = process.env.PORT || 5000;

// Middleware
app.use(cors());
app.use(express.json());
app.use(express.urlencoded({ extended: true }));

// MongoDB connection
const connectDB = async () => {
  try {
    await mongoose.connect(process.env.MONGODB_URI || 'mongodb://localhost:27017/podcast-streaming');
    console.log('Connected to MongoDB');
  } catch (error) {
    console.error('MongoDB connection error:', error);
    process.exit(1);
  }
};

// User Schema
const userSchema = new mongoose.Schema({
  username: { type: String, required: true, unique: true },
  email: { type: String, required: true, unique: true },
  password: { type: String, required: true },
  firstName: { type: String, required: true },
  lastName: { type: String, required: true },
  role: { type: String, enum: ['listener', 'creator', 'admin'], default: 'listener' },
  createdAt: { type: Date, default: Date.now }
});

// Hash password before saving
userSchema.pre('save', async function(next) {
  if (!this.isModified('password')) return next();
  this.password = await bcrypt.hash(this.password, 12);
  next();
});

// Compare password method
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
  coverImage: { type: String, default: null },
  episodes: [{
    title: { type: String, required: true },
    description: { type: String, required: true },
    audioFile: { type: String, required: true },
    duration: { type: Number, required: true },
    episodeNumber: { type: Number, required: true },
    publishDate: { type: Date, default: Date.now },
    isPublished: { type: Boolean, default: true }
  }],
  subscribers: [{ type: mongoose.Schema.Types.ObjectId, ref: 'User' }],
  isActive: { type: Boolean, default: true },
  createdAt: { type: Date, default: Date.now }
});

const Podcast = mongoose.model('Podcast', podcastSchema);

// Auth middleware
const authenticateToken = async (req, res, next) => {
  const authHeader = req.headers['authorization'];
  const token = authHeader && authHeader.split(' ')[1];

  if (!token) {
    return res.status(401).json({ success: false, message: 'Access token required' });
  }

  try {
    const decoded = jwt.verify(token, process.env.JWT_SECRET || 'fallback-secret');
    const user = await User.findById(decoded.userId);
    if (!user) {
      return res.status(401).json({ success: false, message: 'Invalid token' });
    }
    req.user = user;
    next();
  } catch (error) {
    return res.status(403).json({ success: false, message: 'Invalid token' });
  }
};

// Routes

// Health check
app.get('/api/health', (req, res) => {
  res.json({ 
    status: 'OK', 
    message: 'Podcast Streaming API is running',
    timestamp: new Date().toISOString()
  });
});

// Register
app.post('/api/auth/register', async (req, res) => {
  try {
    const { username, email, password, firstName, lastName, role } = req.body;

    // Check if user already exists
    const existingUser = await User.findOne({ 
      $or: [{ email }, { username }] 
    });

    if (existingUser) {
      return res.status(400).json({
        success: false,
        message: 'User with this email or username already exists'
      });
    }

    // Create new user
    const user = new User({
      username,
      email,
      password,
      firstName,
      lastName,
      role: role || 'listener'
    });

    await user.save();

    // Generate token
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
      message: 'Server error during registration'
    });
  }
});

// Login
app.post('/api/auth/login', async (req, res) => {
  try {
    const { email, password } = req.body;

    // Find user
    const user = await User.findOne({ email });
    if (!user) {
      return res.status(401).json({
        success: false,
        message: 'Invalid credentials'
      });
    }

    // Check password
    const isValidPassword = await user.comparePassword(password);
    if (!isValidPassword) {
      return res.status(401).json({
        success: false,
        message: 'Invalid credentials'
      });
    }

    // Generate token
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
      message: 'Server error during login'
    });
  }
});

// Get all podcasts
app.get('/api/podcasts', async (req, res) => {
  try {
    const { category, search, page = 1, limit = 12 } = req.query;
    
    let query = { isActive: true };
    
    if (category) {
      query.category = category;
    }
    
    if (search) {
      query.$or = [
        { title: { $regex: search, $options: 'i' } },
        { description: { $regex: search, $options: 'i' } }
      ];
    }

    const podcasts = await Podcast.find(query)
      .populate('creator', 'username firstName lastName')
      .sort({ createdAt: -1 })
      .limit(parseInt(limit))
      .skip((parseInt(page) - 1) * parseInt(limit));

    const total = await Podcast.countDocuments(query);

    res.json({
      success: true,
      podcasts: podcasts.map(podcast => ({
        ...podcast.toObject(),
        episodeCount: podcast.episodes.length,
        subscriberCount: podcast.subscribers.length
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
      message: 'Server error fetching podcasts'
    });
  }
});

// Get podcast categories
app.get('/api/podcasts/categories', (req, res) => {
  const categories = [
    'Technology', 'Business', 'Education', 'Health & Fitness',
    'News & Politics', 'Entertainment', 'Sports', 'Music',
    'Comedy', 'History', 'Science', 'Arts'
  ];

  res.json({
    success: true,
    categories
  });
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
      message: 'Server error fetching podcast'
    });
  }
});

// Create podcast (requires authentication and creator role)
app.post('/api/podcasts', authenticateToken, async (req, res) => {
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
      message: 'Server error creating podcast'
    });
  }
});

// Subscribe to podcast
app.post('/api/podcasts/:id/subscribe', authenticateToken, async (req, res) => {
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
      message: 'Successfully subscribed to podcast'
    });
  } catch (error) {
    console.error('Subscribe error:', error);
    res.status(500).json({
      success: false,
      message: 'Server error subscribing to podcast'
    });
  }
});

// Error handling middleware
app.use((err, req, res, next) => {
  console.error(err.stack);
  res.status(500).json({
    success: false,
    message: 'Something went wrong!'
  });
});

// 404 handler
app.use('*', (req, res) => {
  res.status(404).json({
    success: false,
    message: 'Route not found'
  });
});

// Start server
const startServer = async () => {
  await connectDB();
  app.listen(PORT, '0.0.0.0', () => {
    console.log(`🎧 Podcast Streaming Server running on port ${PORT}`);
  });
};

startServer().catch(console.error);

module.exports = app;