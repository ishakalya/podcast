const express = require('express');
const mongoose = require('mongoose');
const cors = require('cors');
const bcrypt = require('bcryptjs');
const jwt = require('jsonwebtoken');
const serverless = require('serverless-http');

const app = express();

// Basic middleware
app.use(cors());
app.use(express.json({ limit: '10mb' }));

// In-memory storage for demo (since MongoDB might not be available on Netlify)
let inMemoryUsers = [];
let inMemoryPodcasts = [];
let userIdCounter = 1;
let podcastIdCounter = 1;

// Initialize demo data
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

// Initialize demo data
initializeDemoData();

// Helper functions
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
    const user = inMemoryUsers.find(u => u._id === decoded.userId);

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
    message: '🎧 Podcast Streaming API', 
    status: 'running',
    database: 'In-Memory Demo',
    timestamp: new Date().toISOString()
  });
});

app.get('/health', (req, res) => {
  res.json({ 
    status: 'OK', 
    message: 'API is healthy',
    database: 'In-Memory Mode',
    timestamp: new Date().toISOString()
  });
});

// Register
app.post('/auth/register', async (req, res) => {
  try {
    const { username, email, password, firstName, lastName, role } = req.body;

    if (!username || !email || !password || !firstName || !lastName) {
      return res.status(400).json({
        success: false,
        message: 'All fields are required'
      });
    }

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
      message: 'User created successfully',
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
  } catch (error) {
    console.error('Registration error:', error);
    res.status(500).json({
      success: false,
      message: 'User creation failed'
    });
  }
});

// Login
app.post('/auth/login', async (req, res) => {
  try {
    const { email, password } = req.body;

    const user = inMemoryUsers.find(u => u.email === email);
    if (!user || !await comparePassword(password, user.password)) {
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
app.get('/auth/profile', auth, async (req, res) => {
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
app.get('/podcasts/categories', (req, res) => {
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
app.get('/podcasts', async (req, res) => {
  try {
    const { category, search, page = 1, limit = 12 } = req.query;

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
  } catch (error) {
    console.error('Get podcasts error:', error);
    res.status(500).json({
      success: false,
      message: 'Failed to fetch podcasts'
    });
  }
});

// Get specific podcast
app.get('/podcasts/:id', async (req, res) => {
  try {
    const podcast = inMemoryPodcasts.find(p => p._id === req.params.id && p.isActive);
    
    if (!podcast) {
      return res.status(404).json({
        success: false,
        message: 'Podcast not found'
      });
    }

    res.json({
      success: true,
      podcast: {
        ...podcast,
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
app.post('/podcasts', auth, async (req, res) => {
  try {
    if (req.user.role !== 'creator' && req.user.role !== 'admin') {
      return res.status(403).json({
        success: false,
        message: 'Creator privileges required'
      });
    }

    const { title, description, category } = req.body;

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
      message: 'Podcast created successfully',
      podcast: newPodcast
    });
  } catch (error) {
    console.error('Create podcast error:', error);
    res.status(500).json({
      success: false,
      message: 'Failed to create podcast'
    });
  }
});

// Update podcast
app.put('/podcasts/:id', auth, async (req, res) => {
  try {
    const podcastIndex = inMemoryPodcasts.findIndex(p => p._id === req.params.id);
    
    if (podcastIndex === -1) {
      return res.status(404).json({
        success: false,
        message: 'Podcast not found'
      });
    }

    const podcast = inMemoryPodcasts[podcastIndex];
    
    if (podcast.creator._id !== req.user._id && req.user.role !== 'admin') {
      return res.status(403).json({
        success: false,
        message: 'Not authorized to update this podcast'
      });
    }

    const { title, description, category } = req.body;

    inMemoryPodcasts[podcastIndex] = {
      ...podcast,
      title: title || podcast.title,
      description: description || podcast.description,
      category: category || podcast.category,
      updatedAt: new Date()
    };

    res.json({
      success: true,
      message: 'Podcast updated successfully',
      podcast: inMemoryPodcasts[podcastIndex]
    });
  } catch (error) {
    console.error('Update podcast error:', error);
    res.status(500).json({
      success: false,
      message: 'Failed to update podcast'
    });
  }
});

// Delete podcast
app.delete('/podcasts/:id', auth, async (req, res) => {
  try {
    const podcastIndex = inMemoryPodcasts.findIndex(p => p._id === req.params.id);
    
    if (podcastIndex === -1) {
      return res.status(404).json({
        success: false,
        message: 'Podcast not found'
      });
    }

    const podcast = inMemoryPodcasts[podcastIndex];
    
    if (podcast.creator._id !== req.user._id && req.user.role !== 'admin') {
      return res.status(403).json({
        success: false,
        message: 'Not authorized to delete this podcast'
      });
    }

    // Soft delete
    inMemoryPodcasts[podcastIndex].isActive = false;

    res.json({
      success: true,
      message: 'Podcast deleted successfully'
    });
  } catch (error) {
    console.error('Delete podcast error:', error);
    res.status(500).json({
      success: false,
      message: 'Failed to delete podcast'
    });
  }
});

// Subscribe to podcast
app.post('/podcasts/:id/subscribe', auth, async (req, res) => {
  try {
    const podcastIndex = inMemoryPodcasts.findIndex(p => p._id === req.params.id);
    
    if (podcastIndex === -1) {
      return res.status(404).json({
        success: false,
        message: 'Podcast not found'
      });
    }

    const podcast = inMemoryPodcasts[podcastIndex];
    if (!podcast.subscribers.includes(req.user._id)) {
      podcast.subscribers.push(req.user._id);
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
app.get('/users/subscriptions', auth, async (req, res) => {
  try {
    const subscriptions = inMemoryPodcasts.filter(p => 
      p.subscribers.includes(req.user._id) && p.isActive
    );

    res.json({
      success: true,
      subscriptions: subscriptions.map(p => ({
        ...p,
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

// Error handling
app.use((err, req, res, next) => {
  console.error(err.stack);
  res.status(500).json({
    success: false,
    message: 'Something went wrong!'
  });
});

module.exports.handler = serverless(app);