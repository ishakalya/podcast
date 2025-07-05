# Podcast Streaming Service

## Overview

A comprehensive podcast streaming platform that allows users to discover, listen to, and manage podcasts. Creators can upload and manage their podcast content, while listeners can subscribe, favorite episodes, and track their listening progress.

## System Architecture

### Technology Stack
- **Frontend**: React.js with Context API for state management
- **Backend**: Node.js with Express.js framework  
- **Database**: MongoDB for data persistence
- **Authentication**: JWT tokens with bcrypt password hashing
- **File Storage**: Local file system with multer for uploads

### Key Features
- User authentication and authorization (listeners, creators, admins)
- Podcast discovery and browsing with categories and search
- Audio streaming and playback controls
- Subscription management and favorites
- Creator dashboard for podcast and episode management
- User profiles and listening history
- Responsive design for all devices

## System Components

### Frontend (React)
- Component-based architecture with reusable UI elements
- Context providers for authentication and app state
- Protected routes based on user roles
- Responsive design with modern CSS

### Backend (Express.js)
- RESTful API design with proper HTTP methods
- Middleware for authentication, validation, and error handling
- File upload handling for audio and images
- Database models and relationships

### Database (MongoDB)
- User profiles with roles and preferences
- Podcast metadata and episode information
- Subscription and favorite relationships
- Listening history and progress tracking

## Security & Performance
- JWT-based authentication with secure password hashing
- Role-based access control (listeners, creators, admins)
- File upload validation and size limits
- Rate limiting for API endpoints
- CORS configuration for cross-origin requests

## Deployment Strategy
- Environment-based configuration
- MongoDB database setup
- Static file serving for uploads
- Port configuration for production deployment

## Changelog
- July 05, 2025: Project restructure and clean start for podcast streaming service

## User Preferences
Preferred communication style: Simple, everyday language.