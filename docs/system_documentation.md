# System Documentation

This document provides a complete overview of the AI Trading Dashboard system. It includes a summary of the project, links to detailed documentation, and a complete file tree.

## Table of Contents

* [Project Overview](#project-overview)
* [Backend Documentation](#backend-documentation)
* [Frontend Documentation](#frontend-documentation)
* [File Tree](#file-tree)

## Project Overview

The AI Trading Dashboard is a comprehensive system for developing, testing, and deploying automated trading strategies. It features a modular architecture with a React frontend and a FastAPI backend. The system is designed to be self-hosted and uses local file-based storage for data and strategies.

For a quick start guide and information on how to set up and run the project, please see the [main README file](readme.md).

## Backend Documentation

The backend is built with FastAPI and is responsible for all the core logic of the system. This includes managing trading agents, parsing and executing strategies, running backtests, and interacting with the Binance API.

### Backend Summary

The [backend summary](backend/backend%20summary.md) provides a high-level overview of the core backend functions and classes.

### Backend Workflow

The [backend workflow](backend/backendworkflow.md) document describes the complete backend workflow, from handling API requests to executing trades.

### Data Flow

The [data flow](backend/data%20flow.md) document illustrates the flow of data between the frontend and the backend.

### System Diagram

The [system diagram](backend/diagram.md) provides a visual representation of the system architecture.

### Detailed Backend Documentation

The [backend.md](backend/backend.md) file provides a detailed description of each module and function in the backend.

## Frontend Documentation

The frontend is a React application that provides a user-friendly interface for interacting with the system. It allows users to build and test trading strategies, monitor agent performance, and view backtest results.

The [Folder Setup.md](Folder%20Setup.md) file provides an overview of the frontend and backend project structure.

## File Tree

The [file_tree.md](file_tree.md) file contains a complete tree of all files in the project.
