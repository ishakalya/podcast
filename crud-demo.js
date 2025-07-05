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
app.use(express.json({ limit: '10mb' }));
app.use(express.static('public'));

// In-memory storage for demo when MongoDB is not available
let inMemoryUsers = [];
let inMemoryPodcasts = [];
let userIdCounter = 1;
let podcastIdCounter = 1;

// MongoDB connection
let mongoConnected = false;
mongoose.connect(process.env.MONGODB_URI || 'mongodb://localhost:27017/podcast-streaming', {
  useNewUrlParser: true,
  useUnifiedTopology: true,
  serverSelectionTimeoutMS: 3000,
  maxPoolSize: 10
})
.then(() => {
  console.log('✅ Connected to MongoDB successfully');
  mongoConnected = true;
})
.catch(err => {
  console.log('📄 Running in in-memory demo mode for CRUD operations');
  mongoConnected = false;
  // Populate demo data
  initializeDemoData();
});

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

// Initialize demo data for in-memory mode
function initializeDemoData() {
  // Demo users
  inMemoryUsers = [
    {
      _id: '1',
      username: 'demo_listener',
      email: 'listener@demo.com',
      password: '$2a$12$LQv3c1yqBWVHxkd0LHAkCOYz6TtxMQJqhN8/LewTuFJWiKy0ue.F2', // password123
      firstName: 'Demo',
      lastName: 'Listener',
      role: 'listener',
      createdAt: new Date(),
      updatedAt: new Date()
    },
    {
      _id: '2',
      username: 'demo_creator',
      email: 'creator@demo.com',
      password: '$2a$12$LQv3c1yqBWVHxkd0LHAkCOYz6TtxMQJqhN8/LewTuFJWiKy0ue.F2', // password123
      firstName: 'Demo',
      lastName: 'Creator',
      role: 'creator',
      createdAt: new Date(),
      updatedAt: new Date()
    }
  ];

  // Demo podcasts
  inMemoryPodcasts = [
    {
      _id: '1',
      title: 'Tech Talk Today',
      description: 'Daily discussions about the latest in technology and innovation.',
      creator: {
        _id: '2',
        username: 'demo_creator',
        firstName: 'Demo',
        lastName: 'Creator'
      },
      category: 'Technology',
      episodes: [
        {
          _id: 'ep1',
          title: 'The Future of AI',
          description: 'Exploring artificial intelligence trends and their impact on society.',
          duration: 1800,
          episodeNumber: 1,
          publishDate: new Date()
        }
      ],
      subscribers: ['1'],
      isActive: true,
      createdAt: new Date(),
      updatedAt: new Date()
    },
    {
      _id: '2',
      title: 'Business Insights',
      description: 'Expert analysis of business trends and entrepreneurship.',
      creator: {
        _id: '2',
        username: 'demo_creator',
        firstName: 'Demo',
        lastName: 'Creator'
      },
      category: 'Business',
      episodes: [],
      subscribers: [],
      isActive: true,
      createdAt: new Date(),
      updatedAt: new Date()
    }
  ];

  userIdCounter = 3;
  podcastIdCounter = 3;
}

// Helper functions for in-memory operations
async function hashPassword(password) {
  return await bcrypt.hash(password, 12);
}

async function comparePassword(password, hashedPassword) {
  return await bcrypt.compare(password, hashedPassword);
}

// Auth middleware
const auth = async (req, res, next) => {
  try {
    const token = req.header('Authorization')?.replace('Bearer ', '');
    if (!token) {
      return res.status(401).json({ success: false, message: 'Access denied' });
    }

    const decoded = jwt.verify(token, process.env.JWT_SECRET || 'fallback-secret');
    
    let user;
    if (mongoConnected) {
      user = await User.findById(decoded.userId);
    } else {
      user = inMemoryUsers.find(u => u._id === decoded.userId);
    }

    if (!user) {
      return res.status(401).json({ success: false, message: 'Invalid token' });
    }

    req.user = user;
    next();
  } catch (error) {
    res.status(401).json({ success: false, message: 'Invalid token' });
  }
};

// Routes
app.get('/', (req, res) => {
  res.json({ 
    message: '🎧 Podcast Streaming API with Full CRUD Operations', 
    status: 'running',
    database: mongoConnected ? 'MongoDB' : 'In-Memory Demo',
    timestamp: new Date().toISOString(),
    features: [
      'Create, Read, Update, Delete Users',
      'Create, Read, Update, Delete Podcasts',
      'Episode Management',
      'User Authentication',
      'Subscription Management'
    ]
  });
});

app.get('/api/health', (req, res) => {
  res.json({ 
    status: 'OK', 
    message: 'API is healthy',
    database: mongoConnected ? 'MongoDB Connected' : 'In-Memory Mode',
    timestamp: new Date().toISOString()
  });
});

// ============ USER CRUD OPERATIONS ============

// CREATE USER (Register)
app.post('/api/auth/register', async (req, res) => {
  try {
    const { username, email, password, firstName, lastName, role } = req.body;

    // Validation
    if (!username || !email || !password || !firstName || !lastName) {
      return res.status(400).json({
        success: false,
        message: 'All fields are required'
      });
    }

    if (mongoConnected) {
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
        message: 'User created successfully',
        operation: 'CREATE',
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
    } else {
      // In-memory implementation
      const existingUser = inMemoryUsers.find(u => u.email === email || u.username === username);
      if (existingUser) {
        return res.status(400).json({
          success: false,
          message: 'User already exists with this email or username'
        });
      }

      const hashedPassword = await hashPassword(password);
      const newUser = {
        _id: userIdCounter.toString(),
        username,
        email,
        password: hashedPassword,
        firstName,
        lastName,
        role: role || 'listener',
        createdAt: new Date(),
        updatedAt: new Date()
      };

      inMemoryUsers.push(newUser);
      userIdCounter++;

      const token = jwt.sign(
        { userId: newUser._id },
        process.env.JWT_SECRET || 'fallback-secret',
        { expiresIn: '7d' }
      );

      res.status(201).json({
        success: true,
        message: 'User created successfully (In-Memory)',
        operation: 'CREATE',
        token,
        user: {
          id: newUser._id,
          username: newUser.username,
          email: newUser.email,
          firstName: newUser.firstName,
          lastName: newUser.lastName,
          role: newUser.role
        }
      });
    }
  } catch (error) {
    console.error('Registration error:', error);
    res.status(500).json({
      success: false,
      message: 'User creation failed',
      operation: 'CREATE'
    });
  }
});

// READ USER (Login)
app.post('/api/auth/login', async (req, res) => {
  try {
    const { email, password } = req.body;

    if (mongoConnected) {
      const user = await User.findOne({ email });
      if (!user || !await user.comparePassword(password)) {
        return res.status(401).json({
          success: false,
          message: 'Invalid credentials',
          operation: 'READ'
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
        operation: 'READ',
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
    } else {
      // In-memory implementation
      const user = inMemoryUsers.find(u => u.email === email);
      if (!user || !await comparePassword(password, user.password)) {
        return res.status(401).json({
          success: false,
          message: 'Invalid credentials',
          operation: 'READ'
        });
      }

      const token = jwt.sign(
        { userId: user._id },
        process.env.JWT_SECRET || 'fallback-secret',
        { expiresIn: '7d' }
      );

      res.json({
        success: true,
        message: 'Login successful (In-Memory)',
        operation: 'READ',
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
    }
  } catch (error) {
    console.error('Login error:', error);
    res.status(500).json({
      success: false,
      message: 'Login failed',
      operation: 'READ'
    });
  }
});

// READ USER (Get Profile)
app.get('/api/auth/profile', auth, async (req, res) => {
  res.json({
    success: true,
    message: 'Profile retrieved successfully',
    operation: 'READ',
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

// UPDATE USER
app.put('/api/auth/profile', auth, async (req, res) => {
  try {
    const { firstName, lastName, email } = req.body;
    const userId = req.user._id;

    if (mongoConnected) {
      const user = await User.findByIdAndUpdate(
        userId,
        { firstName, lastName, email },
        { new: true, runValidators: true }
      );

      res.json({
        success: true,
        message: 'Profile updated successfully',
        operation: 'UPDATE',
        user: {
          id: user._id,
          username: user.username,
          email: user.email,
          firstName: user.firstName,
          lastName: user.lastName,
          role: user.role
        }
      });
    } else {
      // In-memory implementation
      const userIndex = inMemoryUsers.findIndex(u => u._id === userId);
      if (userIndex !== -1) {
        inMemoryUsers[userIndex] = {
          ...inMemoryUsers[userIndex],
          firstName: firstName || inMemoryUsers[userIndex].firstName,
          lastName: lastName || inMemoryUsers[userIndex].lastName,
          email: email || inMemoryUsers[userIndex].email,
          updatedAt: new Date()
        };

        res.json({
          success: true,
          message: 'Profile updated successfully (In-Memory)',
          operation: 'UPDATE',
          user: {
            id: inMemoryUsers[userIndex]._id,
            username: inMemoryUsers[userIndex].username,
            email: inMemoryUsers[userIndex].email,
            firstName: inMemoryUsers[userIndex].firstName,
            lastName: inMemoryUsers[userIndex].lastName,
            role: inMemoryUsers[userIndex].role
          }
        });
      } else {
        res.status(404).json({
          success: false,
          message: 'User not found',
          operation: 'UPDATE'
        });
      }
    }
  } catch (error) {
    console.error('Profile update error:', error);
    res.status(500).json({
      success: false,
      message: 'Profile update failed',
      operation: 'UPDATE'
    });
  }
});

// DELETE USER
app.delete('/api/auth/profile', auth, async (req, res) => {
  try {
    const userId = req.user._id;

    if (mongoConnected) {
      await User.findByIdAndDelete(userId);
      res.json({
        success: true,
        message: 'User deleted successfully',
        operation: 'DELETE'
      });
    } else {
      // In-memory implementation
      const userIndex = inMemoryUsers.findIndex(u => u._id === userId);
      if (userIndex !== -1) {
        inMemoryUsers.splice(userIndex, 1);
        res.json({
          success: true,
          message: 'User deleted successfully (In-Memory)',
          operation: 'DELETE'
        });
      } else {
        res.status(404).json({
          success: false,
          message: 'User not found',
          operation: 'DELETE'
        });
      }
    }
  } catch (error) {
    console.error('User deletion error:', error);
    res.status(500).json({
      success: false,
      message: 'User deletion failed',
      operation: 'DELETE'
    });
  }
});

// ============ PODCAST CRUD OPERATIONS ============

// READ PODCASTS (Get all)
app.get('/api/podcasts', async (req, res) => {
  try {
    const { category, search, page = 1, limit = 12 } = req.query;

    if (mongoConnected) {
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
        message: 'Podcasts retrieved successfully',
        operation: 'READ',
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
    } else {
      // In-memory implementation
      let filteredPodcasts = inMemoryPodcasts.filter(p => p.isActive);
      
      if (category) {
        filteredPodcasts = filteredPodcasts.filter(p => p.category === category);
      }
      
      if (search) {
        filteredPodcasts = filteredPodcasts.filter(p => 
          p.title.toLowerCase().includes(search.toLowerCase()) ||
          p.description.toLowerCase().includes(search.toLowerCase())
        );
      }

      const total = filteredPodcasts.length;
      const startIndex = (parseInt(page) - 1) * parseInt(limit);
      const endIndex = startIndex + parseInt(limit);
      const paginatedPodcasts = filteredPodcasts.slice(startIndex, endIndex);

      res.json({
        success: true,
        message: 'Podcasts retrieved successfully (In-Memory)',
        operation: 'READ',
        podcasts: paginatedPodcasts.map(p => ({
          ...p,
          episodeCount: p.episodes.length,
          subscriberCount: p.subscribers.length
        })),
        pagination: {
          currentPage: parseInt(page),
          totalPages: Math.ceil(total / parseInt(limit)),
          totalPodcasts: total
        }
      });
    }
  } catch (error) {
    console.error('Get podcasts error:', error);
    res.status(500).json({
      success: false,
      message: 'Failed to fetch podcasts',
      operation: 'READ'
    });
  }
});

// Get categories (must come before :id route)
app.get('/api/podcasts/categories', (req, res) => {
  res.json({
    success: true,
    message: 'Categories retrieved successfully',
    operation: 'READ',
    categories: [
      'Technology', 'Business', 'Education', 'Health & Fitness',
      'News & Politics', 'Entertainment', 'Sports', 'Music',
      'Comedy', 'History', 'Science', 'Arts'
    ]
  });
});

// READ PODCAST (Get specific)
app.get('/api/podcasts/:id', async (req, res) => {
  try {
    if (mongoConnected) {
      const podcast = await Podcast.findOne({ 
        _id: req.params.id, 
        isActive: true 
      }).populate('creator', 'username firstName lastName');

      if (!podcast) {
        return res.status(404).json({
          success: false,
          message: 'Podcast not found',
          operation: 'READ'
        });
      }

      res.json({
        success: true,
        message: 'Podcast retrieved successfully',
        operation: 'READ',
        podcast: {
          ...podcast.toObject(),
          episodeCount: podcast.episodes.length,
          subscriberCount: podcast.subscribers.length
        }
      });
    } else {
      // In-memory implementation
      const podcast = inMemoryPodcasts.find(p => p._id === req.params.id && p.isActive);
      
      if (!podcast) {
        return res.status(404).json({
          success: false,
          message: 'Podcast not found',
          operation: 'READ'
        });
      }

      res.json({
        success: true,
        message: 'Podcast retrieved successfully (In-Memory)',
        operation: 'READ',
        podcast: {
          ...podcast,
          episodeCount: podcast.episodes.length,
          subscriberCount: podcast.subscribers.length
        }
      });
    }
  } catch (error) {
    console.error('Get podcast error:', error);
    res.status(500).json({
      success: false,
      message: 'Failed to fetch podcast',
      operation: 'READ'
    });
  }
});

// CREATE PODCAST
app.post('/api/podcasts', auth, async (req, res) => {
  try {
    if (req.user.role !== 'creator' && req.user.role !== 'admin') {
      return res.status(403).json({
        success: false,
        message: 'Creator privileges required',
        operation: 'CREATE'
      });
    }

    const { title, description, category } = req.body;

    if (mongoConnected) {
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
        operation: 'CREATE',
        podcast
      });
    } else {
      // In-memory implementation
      const newPodcast = {
        _id: podcastIdCounter.toString(),
        title,
        description,
        category,
        creator: {
          _id: req.user._id,
          username: req.user.username,
          firstName: req.user.firstName,
          lastName: req.user.lastName
        },
        episodes: [],
        subscribers: [],
        isActive: true,
        createdAt: new Date(),
        updatedAt: new Date()
      };

      inMemoryPodcasts.push(newPodcast);
      podcastIdCounter++;

      res.status(201).json({
        success: true,
        message: 'Podcast created successfully (In-Memory)',
        operation: 'CREATE',
        podcast: newPodcast
      });
    }
  } catch (error) {
    console.error('Create podcast error:', error);
    res.status(500).json({
      success: false,
      message: 'Failed to create podcast',
      operation: 'CREATE'
    });
  }
});

// UPDATE PODCAST
app.put('/api/podcasts/:id', auth, async (req, res) => {
  try {
    const { title, description, category } = req.body;

    if (mongoConnected) {
      const podcast = await Podcast.findById(req.params.id);

      if (!podcast) {
        return res.status(404).json({
          success: false,
          message: 'Podcast not found',
          operation: 'UPDATE'
        });
      }

      if (podcast.creator.toString() !== req.user._id && req.user.role !== 'admin') {
        return res.status(403).json({
          success: false,
          message: 'Not authorized to update this podcast',
          operation: 'UPDATE'
        });
      }

      podcast.title = title || podcast.title;
      podcast.description = description || podcast.description;
      podcast.category = category || podcast.category;

      await podcast.save();
      await podcast.populate('creator', 'username firstName lastName');

      res.json({
        success: true,
        message: 'Podcast updated successfully',
        operation: 'UPDATE',
        podcast
      });
    } else {
      // In-memory implementation
      const podcastIndex = inMemoryPodcasts.findIndex(p => p._id === req.params.id);
      
      if (podcastIndex === -1) {
        return res.status(404).json({
          success: false,
          message: 'Podcast not found',
          operation: 'UPDATE'
        });
      }

      const podcast = inMemoryPodcasts[podcastIndex];
      
      if (podcast.creator._id !== req.user._id && req.user.role !== 'admin') {
        return res.status(403).json({
          success: false,
          message: 'Not authorized to update this podcast',
          operation: 'UPDATE'
        });
      }

      inMemoryPodcasts[podcastIndex] = {
        ...podcast,
        title: title || podcast.title,
        description: description || podcast.description,
        category: category || podcast.category,
        updatedAt: new Date()
      };

      res.json({
        success: true,
        message: 'Podcast updated successfully (In-Memory)',
        operation: 'UPDATE',
        podcast: inMemoryPodcasts[podcastIndex]
      });
    }
  } catch (error) {
    console.error('Update podcast error:', error);
    res.status(500).json({
      success: false,
      message: 'Failed to update podcast',
      operation: 'UPDATE'
    });
  }
});

// DELETE PODCAST
app.delete('/api/podcasts/:id', auth, async (req, res) => {
  try {
    if (mongoConnected) {
      const podcast = await Podcast.findById(req.params.id);

      if (!podcast) {
        return res.status(404).json({
          success: false,
          message: 'Podcast not found',
          operation: 'DELETE'
        });
      }

      if (podcast.creator.toString() !== req.user._id && req.user.role !== 'admin') {
        return res.status(403).json({
          success: false,
          message: 'Not authorized to delete this podcast',
          operation: 'DELETE'
        });
      }

      // Soft delete
      podcast.isActive = false;
      await podcast.save();

      res.json({
        success: true,
        message: 'Podcast deleted successfully',
        operation: 'DELETE'
      });
    } else {
      // In-memory implementation
      const podcastIndex = inMemoryPodcasts.findIndex(p => p._id === req.params.id);
      
      if (podcastIndex === -1) {
        return res.status(404).json({
          success: false,
          message: 'Podcast not found',
          operation: 'DELETE'
        });
      }

      const podcast = inMemoryPodcasts[podcastIndex];
      
      if (podcast.creator._id !== req.user._id && req.user.role !== 'admin') {
        return res.status(403).json({
          success: false,
          message: 'Not authorized to delete this podcast',
          operation: 'DELETE'
        });
      }

      // Soft delete
      inMemoryPodcasts[podcastIndex].isActive = false;

      res.json({
        success: true,
        message: 'Podcast deleted successfully (In-Memory)',
        operation: 'DELETE'
      });
    }
  } catch (error) {
    console.error('Delete podcast error:', error);
    res.status(500).json({
      success: false,
      message: 'Failed to delete podcast',
      operation: 'DELETE'
    });
  }
});

// ============ ADDITIONAL CRUD OPERATIONS ============

// Subscribe to podcast (UPDATE operation)
app.post('/api/podcasts/:id/subscribe', auth, async (req, res) => {
  try {
    if (mongoConnected) {
      const podcast = await Podcast.findById(req.params.id);
      
      if (!podcast) {
        return res.status(404).json({
          success: false,
          message: 'Podcast not found',
          operation: 'UPDATE'
        });
      }

      if (!podcast.subscribers.includes(req.user._id)) {
        podcast.subscribers.push(req.user._id);
        await podcast.save();
      }

      res.json({
        success: true,
        message: 'Successfully subscribed',
        operation: 'UPDATE'
      });
    } else {
      // In-memory implementation
      const podcastIndex = inMemoryPodcasts.findIndex(p => p._id === req.params.id);
      
      if (podcastIndex === -1) {
        return res.status(404).json({
          success: false,
          message: 'Podcast not found',
          operation: 'UPDATE'
        });
      }

      const podcast = inMemoryPodcasts[podcastIndex];
      if (!podcast.subscribers.includes(req.user._id)) {
        podcast.subscribers.push(req.user._id);
      }

      res.json({
        success: true,
        message: 'Successfully subscribed (In-Memory)',
        operation: 'UPDATE'
      });
    }
  } catch (error) {
    console.error('Subscribe error:', error);
    res.status(500).json({
      success: false,
      message: 'Failed to subscribe',
      operation: 'UPDATE'
    });
  }
});

// Get user subscriptions (READ operation)
app.get('/api/users/subscriptions', auth, async (req, res) => {
  try {
    if (mongoConnected) {
      const podcasts = await Podcast.find({
        subscribers: req.user._id,
        isActive: true
      }).populate('creator', 'username firstName lastName');

      res.json({
        success: true,
        message: 'Subscriptions retrieved successfully',
        operation: 'READ',
        subscriptions: podcasts.map(p => ({
          ...p.toObject(),
          episodeCount: p.episodes.length,
          subscriberCount: p.subscribers.length
        }))
      });
    } else {
      // In-memory implementation
      const subscriptions = inMemoryPodcasts.filter(p => 
        p.subscribers.includes(req.user._id) && p.isActive
      );

      res.json({
        success: true,
        message: 'Subscriptions retrieved successfully (In-Memory)',
        operation: 'READ',
        subscriptions: subscriptions.map(p => ({
          ...p,
          episodeCount: p.episodes.length,
          subscriberCount: p.subscribers.length
        }))
      });
    }
  } catch (error) {
    console.error('Get subscriptions error:', error);
    res.status(500).json({
      success: false,
      message: 'Failed to fetch subscriptions',
      operation: 'READ'
    });
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

// 404 handler
app.use((req, res) => {
  res.status(404).json({
    success: false,
    message: 'Route not found'
  });
});

// Start server
app.listen(PORT, '0.0.0.0', () => {
  console.log(`🎧 Podcast Streaming Server with Full CRUD Operations running on port ${PORT}`);
  console.log(`📊 Database Mode: ${mongoConnected ? 'MongoDB' : 'In-Memory Demo'}`);
  console.log(`🔗 API Documentation: http://localhost:${PORT}/`);
});

module.exports = app;