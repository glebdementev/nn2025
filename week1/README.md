# 🚢 Titanic Dataset Visualization

A comprehensive TypeScript-based data visualization tool for analyzing the famous Titanic passenger dataset.

## 📋 Overview

This project provides detailed statistical analysis and interactive visualizations of the Titanic passenger dataset, including:

- **Survival Analysis**: Breakdown by class, gender, and family status
- **Demographic Insights**: Age and fare distributions
- **Interactive Dashboard**: Modern HTML interface with embedded SVG charts
- **Statistical Computations**: Custom implementations without external dependencies

## 🚀 Quick Start

### Option 1: View Live Demo (Recommended)
**🌐 [View Live Demo on GitHub Pages](https://glebdementev.github.io/nn2025/)**

The complete interactive visualization is available online with all charts embedded and ready to explore!

### Option 2: Run Locally

#### Prerequisites
- Node.js (v16 or higher)
- npm

#### Installation & Usage

```bash
# Install dependencies
npm install

# Run the complete visualization analysis
npm run viz

# Generate visualizations and open dashboard
npm run dashboard

# Or run individual commands
npm run build    # Compile TypeScript
npm run start    # Run analysis and generate charts
```

## 📊 Generated Outputs

The script generates several visualization files:

1. **GitHub Pages Compatible**:
   - `index.html` - Single-page interactive dashboard with embedded SVG charts (GitHub Pages ready)

2. **Individual SVG Charts**:
   - `survival-by-class.svg` - Survival rates by passenger class
   - `gender-survival.svg` - Survival comparison between male/female passengers
   - `age-distribution.svg` - Age distribution histogram
   - `fare-distribution.svg` - Fare distribution by price ranges

3. **Standalone Dashboard**:
   - `titanic-dashboard.html` - Alternative dashboard with external SVG references

4. **Console Output**: Detailed statistical analysis printed to terminal

### 🌐 GitHub Pages Deployment

The `index.html` file is specifically designed for GitHub Pages deployment:

- **Self-Contained**: All SVG charts are embedded directly in the HTML
- **No External Dependencies**: Works without separate SVG files
- **Responsive Design**: Adapts to all screen sizes
- **Modern UI**: Professional gradient styling and animations
- **Fast Loading**: Optimized for GitHub Pages hosting

#### Deployment Steps:
1. Push your repository to GitHub
2. Go to repository Settings → Pages
3. Set source to "Deploy from a branch"
4. Select "main" branch and "/ (root)" folder
5. Your visualization will be live at `https://yourusername.github.io/repositoryname/`

## 🎯 Key Insights Revealed

- **Class Impact**: First-class passengers had a 63.0% survival rate vs 24.2% for third-class
- **Gender Disparity**: Women survived at 74.2% rate compared to 18.9% for men
- **Family Effect**: Passengers traveling with family had better survival rates (50.6% vs 30.4%)
- **Age Demographics**: Median age was 28 years, with 177 passengers missing age data

## 🏗️ Technical Architecture

### TypeScript Features Used
- **ES Modules**: Modern module system with proper imports
- **Strong Typing**: Custom interfaces for data structures
- **Object-Oriented Design**: Modular class-based architecture
- **File I/O**: Native Node.js filesystem operations

### Visualization Approach
- **Pure SVG Generation**: No external charting libraries
- **Responsive Design**: Modern CSS Grid and Flexbox
- **Statistical Computing**: Custom implementations of mean, median, and binning
- **Data Parsing**: Robust CSV parsing with quote handling

## 📁 Project Structure

```
nn2025/
├── src/
│   └── titanic-viz.ts          # Main visualization script
├── data/
│   └── train.csv               # Titanic dataset (gitignored)
├── dist/                       # Compiled JavaScript
├── *.svg                       # Generated chart files
├── titanic-dashboard.html      # Interactive dashboard
├── package.json               # Dependencies and scripts
├── tsconfig.json              # TypeScript configuration
└── .gitignore                 # Excludes data files from repo
```

## 🔧 Configuration

The script includes several configurable parameters:

- **Age Bins**: 16 bins from 0-80 years
- **Fare Ranges**: 5 predefined price brackets
- **Chart Dimensions**: 600x400px base size with responsive scaling
- **Color Scheme**: Professional blue/orange/green palette

## 🎨 Visualization Features

### Charts
- **Bar Charts**: Class and gender survival rates
- **Histograms**: Age and fare distributions
- **Statistical Labels**: Counts, percentages, and totals
- **Professional Styling**: Consistent fonts, colors, and spacing

### Dashboard
- **Responsive Grid Layout**: Adapts to different screen sizes
- **Modern UI Design**: Gradient backgrounds and card-based layout
- **Key Statistics**: Prominent display of critical numbers
- **Insights Section**: Automated generation of key findings

## 🛡️ Data Privacy

- **Local Processing**: All analysis runs locally, no external API calls
- **Git Exclusion**: Data files are automatically gitignored
- **No Remote Storage**: Charts and analysis remain on your machine

## 🔍 Dataset Information

- **Source**: Kaggle Titanic Competition Dataset
- **Size**: 891 passenger records
- **Features**: 12 attributes including demographics, ticket info, and survival status
- **Quality**: Some missing values in Age (177 records) and Cabin data

## 📈 Performance

- **Fast Execution**: Completes full analysis in < 2 seconds
- **Memory Efficient**: Processes 891 records with minimal overhead
- **Scalable Design**: Architecture supports larger datasets

## 🤝 Contributing

This is a learning project demonstrating TypeScript data visualization capabilities. The code is designed to be:

- **Educational**: Clear, well-commented implementation
- **Extensible**: Easy to add new chart types or analysis
- **Production-Ready**: Follows TypeScript best practices

---

*Generated with ❤️ using TypeScript and modern web technologies*
