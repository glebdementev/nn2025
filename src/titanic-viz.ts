import * as fs from 'fs';
import * as path from 'path';
import { fileURLToPath } from 'url';

const __filename = fileURLToPath(import.meta.url);
const __dirname = path.dirname(__filename);

interface TitanicPassenger {
  PassengerId: number;
  Survived: number;
  Pclass: number;
  Name: string;
  Sex: string;
  Age: number | null;
  SibSp: number;
  Parch: number;
  Ticket: string;
  Fare: number | null;
  Cabin: string;
  Embarked: string;
}

class TitanicVisualizer {
  private data: TitanicPassenger[] = [];
  
  constructor() {
    this.loadData();
  }

  private loadData(): void {
    try {
      const csvPath = path.join(__dirname, '../data/train.csv');
      const csvContent = fs.readFileSync(csvPath, 'utf-8');
      
      // Parse CSV manually since we're in Node.js environment
      const lines = csvContent.split('\n');
      const headers = lines[0].split(',');
      
      for (let i = 1; i < lines.length; i++) {
        if (lines[i].trim()) {
          const values = this.parseCSVLine(lines[i]);
          if (values.length >= headers.length) {
            const passenger: TitanicPassenger = {
              PassengerId: parseInt(values[0]),
              Survived: parseInt(values[1]),
              Pclass: parseInt(values[2]),
              Name: values[3].replace(/"/g, ''),
              Sex: values[4],
              Age: values[5] ? parseFloat(values[5]) : null,
              SibSp: parseInt(values[6]),
              Parch: parseInt(values[7]),
              Ticket: values[8],
              Fare: values[9] ? parseFloat(values[9]) : null,
              Cabin: values[10] || '',
              Embarked: values[11] || ''
            };
            this.data.push(passenger);
          }
        }
      }
      
      console.log(`Loaded ${this.data.length} passengers`);
      this.generateVisualizations();
    } catch (error) {
      console.error('Error loading data:', error);
    }
  }

  private parseCSVLine(line: string): string[] {
    const result: string[] = [];
    let current = '';
    let inQuotes = false;
    
    for (let i = 0; i < line.length; i++) {
      const char = line[i];
      
      if (char === '"') {
        inQuotes = !inQuotes;
      } else if (char === ',' && !inQuotes) {
        result.push(current);
        current = '';
      } else {
        current += char;
      }
    }
    
    result.push(current);
    return result;
  }

  private generateVisualizations(): void {
    console.log('\n=== TITANIC DATASET ANALYSIS ===\n');
    
    this.basicStatistics();
    this.survivalAnalysis();
    this.ageDistribution();
    this.classAnalysis();
    this.genderAnalysis();
    this.fareAnalysis();
    this.embarkationAnalysis();
    this.familyAnalysis();
    this.generateSVGCharts();
    this.createInteractiveHTML();
  }

  private basicStatistics(): void {
    console.log('📊 BASIC STATISTICS');
    console.log('==================');
    console.log(`Total passengers: ${this.data.length}`);
    console.log(`Survivors: ${this.data.filter(p => p.Survived === 1).length}`);
    console.log(`Non-survivors: ${this.data.filter(p => p.Survived === 0).length}`);
    console.log(`Overall survival rate: ${(this.data.filter(p => p.Survived === 1).length / this.data.length * 100).toFixed(1)}%\n`);
  }

  private survivalAnalysis(): void {
    console.log('🛟 SURVIVAL ANALYSIS');
    console.log('===================');
    
    const survived = this.data.filter(p => p.Survived === 1).length;
    const total = this.data.length;
    const survivalRate = (survived / total * 100).toFixed(1);
    
    console.log(`Survival Rate: ${survivalRate}%`);
    console.log('Survival by Class:');
    
    for (let pclass = 1; pclass <= 3; pclass++) {
      const classPassengers = this.data.filter(p => p.Pclass === pclass);
      const classSurvivors = classPassengers.filter(p => p.Survived === 1);
      const classRate = (classSurvivors.length / classPassengers.length * 100).toFixed(1);
      console.log(`  Class ${pclass}: ${classRate}% (${classSurvivors.length}/${classPassengers.length})`);
    }
    console.log();
  }

  private ageDistribution(): void {
    console.log('👥 AGE DISTRIBUTION');
    console.log('==================');
    
    const agesWithData = this.data.filter(p => p.Age !== null).map(p => p.Age!);
    const avgAge = this.mean(agesWithData);
    const medianAge = this.median(agesWithData);
    const minAge = Math.min(...agesWithData);
    const maxAge = Math.max(...agesWithData);
    
    console.log(`Average age: ${avgAge.toFixed(1)} years`);
    console.log(`Median age: ${medianAge} years`);
    console.log(`Age range: ${minAge} - ${maxAge} years`);
    console.log(`Missing age data: ${this.data.filter(p => p.Age === null).length} passengers\n`);
  }

  private classAnalysis(): void {
    console.log('🎭 CLASS ANALYSIS');
    console.log('================');
    
    for (let pclass = 1; pclass <= 3; pclass++) {
      const classPassengers = this.data.filter(p => p.Pclass === pclass);
      const percentage = (classPassengers.length / this.data.length * 100).toFixed(1);
      console.log(`Class ${pclass}: ${classPassengers.length} passengers (${percentage}%)`);
    }
    console.log();
  }

  private genderAnalysis(): void {
    console.log('👫 GENDER ANALYSIS');
    console.log('=================');
    
    const malePassengers = this.data.filter(p => p.Sex === 'male');
    const femalePassengers = this.data.filter(p => p.Sex === 'female');
    
    const maleSurvivalRate = (malePassengers.filter(p => p.Survived === 1).length / malePassengers.length * 100).toFixed(1);
    const femaleSurvivalRate = (femalePassengers.filter(p => p.Survived === 1).length / femalePassengers.length * 100).toFixed(1);
    
    console.log(`Male passengers: ${malePassengers.length} (survival rate: ${maleSurvivalRate}%)`);
    console.log(`Female passengers: ${femalePassengers.length} (survival rate: ${femaleSurvivalRate}%)`);
    console.log();
  }

  private fareAnalysis(): void {
    console.log('💰 FARE ANALYSIS');
    console.log('===============');
    
    const faresWithData = this.data.filter(p => p.Fare !== null).map(p => p.Fare!);
    const avgFare = this.mean(faresWithData);
    const medianFare = this.median(faresWithData);
    const minFare = Math.min(...faresWithData);
    const maxFare = Math.max(...faresWithData);
    
    console.log(`Average fare: $${avgFare.toFixed(2)}`);
    console.log(`Median fare: $${medianFare.toFixed(2)}`);
    console.log(`Fare range: $${minFare.toFixed(2)} - $${maxFare.toFixed(2)}`);
    console.log();
  }

  private embarkationAnalysis(): void {
    console.log('🚢 EMBARKATION ANALYSIS');
    console.log('======================');
    
    const embarkationCounts: { [key: string]: number } = {};
    this.data.filter(p => p.Embarked).forEach(p => {
      embarkationCounts[p.Embarked] = (embarkationCounts[p.Embarked] || 0) + 1;
    });

    const embarkationNames: { [key: string]: string } = {
      'C': 'Cherbourg',
      'Q': 'Queenstown',
      'S': 'Southampton'
    };

    Object.entries(embarkationCounts).forEach(([port, count]) => {
      const percentage = (count / this.data.length * 100).toFixed(1);
      const fullName = embarkationNames[port] || port;
      console.log(`${fullName} (${port}): ${count} passengers (${percentage}%)`);
    });
    console.log();
  }

  private familyAnalysis(): void {
    console.log('👨‍👩‍👧‍👦 FAMILY ANALYSIS');
    console.log('==================');
    
    const soloTravelers = this.data.filter(p => p.SibSp === 0 && p.Parch === 0);
    const withFamily = this.data.filter(p => p.SibSp > 0 || p.Parch > 0);
    
    const soloSurvivalRate = (soloTravelers.filter(p => p.Survived === 1).length / soloTravelers.length * 100).toFixed(1);
    const familySurvivalRate = (withFamily.filter(p => p.Survived === 1).length / withFamily.length * 100).toFixed(1);
    
    console.log(`Solo travelers: ${soloTravelers.length} (survival rate: ${soloSurvivalRate}%)`);
    console.log(`Traveling with family: ${withFamily.length} (survival rate: ${familySurvivalRate}%)`);
    console.log();
  }

  private generateSVGCharts(): void {
    console.log('📈 GENERATING SVG CHARTS');
    console.log('========================');
    
    this.createSurvivalByClassChart();
    this.createAgeDistributionChart();
    this.createFareDistributionChart();
    this.createGenderSurvivalChart();
    
    console.log('Charts saved as SVG files!\n');
  }

  private createSurvivalByClassChart(): void {
    const margin = { top: 50, right: 30, bottom: 60, left: 60 };
    const width = 600 - margin.left - margin.right;
    const height = 400 - margin.top - margin.bottom;

    // Calculate survival rates by class
    const classData = [1, 2, 3].map(pclass => {
      const classPassengers = this.data.filter(p => p.Pclass === pclass);
      const survivors = classPassengers.filter(p => p.Survived === 1).length;
      return {
        class: `Class ${pclass}`,
        rate: (survivors / classPassengers.length) * 100,
        survivors,
        total: classPassengers.length
      };
    });

    const svg = `
<svg width="${width + margin.left + margin.right}" height="${height + margin.top + margin.bottom}" xmlns="http://www.w3.org/2000/svg">
  <defs>
    <style>
      .title { font-family: Arial, sans-serif; font-size: 18px; font-weight: bold; text-anchor: middle; }
      .axis-label { font-family: Arial, sans-serif; font-size: 14px; text-anchor: middle; }
      .bar-label { font-family: Arial, sans-serif; font-size: 12px; font-weight: bold; text-anchor: middle; fill: white; }
      .tick-label { font-family: Arial, sans-serif; font-size: 11px; text-anchor: middle; }
      .y-tick-label { font-family: Arial, sans-serif; font-size: 11px; text-anchor: end; }
    </style>
  </defs>
  <g transform="translate(${margin.left},${margin.top})">
    
    <!-- Chart title -->
    <text x="${width / 2}" y="-20" class="title">Survival Rate by Passenger Class</text>
    
    <!-- Bars -->
    ${classData.map((d, i) => `
      <rect x="${i * (width / 3) + 30}" y="${height - (d.rate / 100) * height}" 
            width="${width / 3 - 60}" height="${(d.rate / 100) * height}"
            fill="${['#1f77b4', '#ff7f0e', '#2ca02c'][i]}" stroke="#333" stroke-width="1"/>
      
      <!-- Bar value labels -->
      <text x="${i * (width / 3) + width / 6}" y="${height - (d.rate / 100) * height + 20}"
            class="bar-label">${d.rate.toFixed(1)}%</text>
      
      <!-- Class labels -->
      <text x="${i * (width / 3) + width / 6}" y="${height + 25}" class="tick-label">${d.class}</text>
      
      <!-- Additional info -->
      <text x="${i * (width / 3) + width / 6}" y="${height + 40}" class="tick-label" font-size="10px">
        ${d.survivors}/${d.total} survived
      </text>
    `).join('')}
    
    <!-- Y-axis -->
    <line x1="0" y1="0" x2="0" y2="${height}" stroke="#333" stroke-width="2"/>
    
    <!-- X-axis -->
    <line x1="0" y1="${height}" x2="${width}" y2="${height}" stroke="#333" stroke-width="2"/>
    
    <!-- Y-axis ticks and labels -->
    ${[0, 20, 40, 60, 80, 100].map(tick => `
      <line x1="-5" y1="${height - (tick / 100) * height}" x2="0" y2="${height - (tick / 100) * height}" stroke="#333"/>
      <text x="-10" y="${height - (tick / 100) * height + 4}" class="y-tick-label">${tick}%</text>
    `).join('')}
    
    <!-- Y-axis title -->
    <text transform="rotate(-90)" x="${-height / 2}" y="-35" class="axis-label">Survival Rate</text>
    
    <!-- X-axis title -->
    <text x="${width / 2}" y="${height + 55}" class="axis-label">Passenger Class</text>
  </g>
</svg>`;

    fs.writeFileSync(path.join(__dirname, '../survival-by-class.svg'), svg);
  }

  private createAgeDistributionChart(): void {
    const margin = { top: 50, right: 30, bottom: 60, left: 60 };
    const width = 600 - margin.left - margin.right;
    const height = 400 - margin.top - margin.bottom;

    // Create age bins
    const agesWithData = this.data.filter(p => p.Age !== null).map(p => p.Age!);
    const bins = this.createBins(agesWithData, 0, 80, 16);
    
    const maxCount = Math.max(...bins.map(b => b.count));
    const binWidth = width / bins.length;

    const svg = `
<svg width="${width + margin.left + margin.right}" height="${height + margin.top + margin.bottom}" xmlns="http://www.w3.org/2000/svg">
  <defs>
    <style>
      .title { font-family: Arial, sans-serif; font-size: 18px; font-weight: bold; text-anchor: middle; }
      .axis-label { font-family: Arial, sans-serif; font-size: 14px; text-anchor: middle; }
      .count-label { font-family: Arial, sans-serif; font-size: 10px; text-anchor: middle; }
      .tick-label { font-family: Arial, sans-serif; font-size: 10px; text-anchor: middle; }
    </style>
  </defs>
  <g transform="translate(${margin.left},${margin.top})">
    
    <!-- Chart title -->
    <text x="${width / 2}" y="-20" class="title">Age Distribution of Passengers</text>
    
    <!-- Bars -->
    ${bins.map((bin, i) => `
      <rect x="${i * binWidth}" y="${height - (bin.count / maxCount) * height}" 
            width="${binWidth - 1}" height="${(bin.count / maxCount) * height}"
            fill="#69b3a2" stroke="#333" stroke-width="0.5"/>
      
      ${bin.count > 0 ? `
        <text x="${i * binWidth + binWidth / 2}" y="${height - (bin.count / maxCount) * height - 5}"
              class="count-label">${bin.count}</text>
      ` : ''}
    `).join('')}
    
    <!-- Axes -->
    <line x1="0" y1="0" x2="0" y2="${height}" stroke="#333" stroke-width="2"/>
    <line x1="0" y1="${height}" x2="${width}" y2="${height}" stroke="#333" stroke-width="2"/>
    
    <!-- X-axis labels -->
    ${bins.map((bin, i) => i % 2 === 0 ? `
      <text x="${i * binWidth + binWidth / 2}" y="${height + 20}" class="tick-label">
        ${Math.round(bin.min)}-${Math.round(bin.max)}
      </text>
    ` : '').join('')}
    
    <!-- Chart titles -->
    <text x="${width / 2}" y="${height + 45}" class="axis-label">Age (years)</text>
    <text transform="rotate(-90)" x="${-height / 2}" y="-35" class="axis-label">Number of Passengers</text>
  </g>
</svg>`;

    fs.writeFileSync(path.join(__dirname, '../age-distribution.svg'), svg);
  }

  private createFareDistributionChart(): void {
    const margin = { top: 50, right: 30, bottom: 60, left: 60 };
    const width = 600 - margin.left - margin.right;
    const height = 400 - margin.top - margin.bottom;

    const fareRanges = [
      { min: 0, max: 10, label: '$0-10' },
      { min: 10, max: 25, label: '$10-25' },
      { min: 25, max: 50, label: '$25-50' },
      { min: 50, max: 100, label: '$50-100' },
      { min: 100, max: 1000, label: '$100+' }
    ];

    const fareData = fareRanges.map(range => {
      const count = this.data.filter(p => 
        p.Fare !== null && p.Fare >= range.min && p.Fare < range.max
      ).length;
      return { ...range, count };
    });

    const maxCount = Math.max(...fareData.map(d => d.count));
    const barWidth = width / fareData.length;

    const svg = `
<svg width="${width + margin.left + margin.right}" height="${height + margin.top + margin.bottom}" xmlns="http://www.w3.org/2000/svg">
  <defs>
    <style>
      .title { font-family: Arial, sans-serif; font-size: 18px; font-weight: bold; text-anchor: middle; }
      .axis-label { font-family: Arial, sans-serif; font-size: 14px; text-anchor: middle; }
      .count-label { font-family: Arial, sans-serif; font-size: 12px; font-weight: bold; text-anchor: middle; fill: white; }
      .tick-label { font-family: Arial, sans-serif; font-size: 11px; text-anchor: middle; }
    </style>
  </defs>
  <g transform="translate(${margin.left},${margin.top})">
    
    <!-- Chart title -->
    <text x="${width / 2}" y="-20" class="title">Fare Distribution</text>
    
    <!-- Bars -->
    ${fareData.map((d, i) => `
      <rect x="${i * barWidth + 20}" y="${height - (d.count / maxCount) * height}" 
            width="${barWidth - 40}" height="${(d.count / maxCount) * height}"
            fill="#ff7f0e" stroke="#333" stroke-width="1"/>
      
      <text x="${i * barWidth + barWidth / 2}" y="${height - (d.count / maxCount) * height + 20}"
            class="count-label">${d.count}</text>
      
      <text x="${i * barWidth + barWidth / 2}" y="${height + 25}" class="tick-label">${d.label}</text>
    `).join('')}
    
    <!-- Axes -->
    <line x1="0" y1="0" x2="0" y2="${height}" stroke="#333" stroke-width="2"/>
    <line x1="0" y1="${height}" x2="${width}" y2="${height}" stroke="#333" stroke-width="2"/>
    
    <!-- Axis titles -->
    <text x="${width / 2}" y="${height + 45}" class="axis-label">Fare Range</text>
    <text transform="rotate(-90)" x="${-height / 2}" y="-35" class="axis-label">Number of Passengers</text>
  </g>
</svg>`;

    fs.writeFileSync(path.join(__dirname, '../fare-distribution.svg'), svg);
  }

  private createGenderSurvivalChart(): void {
    const margin = { top: 50, right: 30, bottom: 60, left: 60 };
    const width = 600 - margin.left - margin.right;
    const height = 400 - margin.top - margin.bottom;

    const genderData = ['male', 'female'].map(gender => {
      const genderPassengers = this.data.filter(p => p.Sex === gender);
      const survivors = genderPassengers.filter(p => p.Survived === 1).length;
      return {
        gender: gender.charAt(0).toUpperCase() + gender.slice(1),
        rate: (survivors / genderPassengers.length) * 100,
        survivors,
        total: genderPassengers.length
      };
    });

    const svg = `
<svg width="${width + margin.left + margin.right}" height="${height + margin.top + margin.bottom}" xmlns="http://www.w3.org/2000/svg">
  <defs>
    <style>
      .title { font-family: Arial, sans-serif; font-size: 18px; font-weight: bold; text-anchor: middle; }
      .axis-label { font-family: Arial, sans-serif; font-size: 14px; text-anchor: middle; }
      .bar-label { font-family: Arial, sans-serif; font-size: 12px; font-weight: bold; text-anchor: middle; fill: white; }
      .tick-label { font-family: Arial, sans-serif; font-size: 11px; text-anchor: middle; }
      .y-tick-label { font-family: Arial, sans-serif; font-size: 11px; text-anchor: end; }
    </style>
  </defs>
  <g transform="translate(${margin.left},${margin.top})">
    
    <!-- Chart title -->
    <text x="${width / 2}" y="-20" class="title">Survival Rate by Gender</text>
    
    <!-- Bars -->
    ${genderData.map((d, i) => `
      <rect x="${i * (width / 2) + 60}" y="${height - (d.rate / 100) * height}" 
            width="${width / 2 - 120}" height="${(d.rate / 100) * height}"
            fill="${i === 0 ? '#3498db' : '#e74c3c'}" stroke="#333" stroke-width="1"/>
      
      <text x="${i * (width / 2) + width / 4}" y="${height - (d.rate / 100) * height + 20}"
            class="bar-label">${d.rate.toFixed(1)}%</text>
      
      <text x="${i * (width / 2) + width / 4}" y="${height + 25}" class="tick-label">${d.gender}</text>
      
      <text x="${i * (width / 2) + width / 4}" y="${height + 40}" class="tick-label" font-size="10px">
        ${d.survivors}/${d.total} survived
      </text>
    `).join('')}
    
    <!-- Y-axis -->
    <line x1="0" y1="0" x2="0" y2="${height}" stroke="#333" stroke-width="2"/>
    
    <!-- X-axis -->
    <line x1="0" y1="${height}" x2="${width}" y2="${height}" stroke="#333" stroke-width="2"/>
    
    <!-- Y-axis ticks and labels -->
    ${[0, 20, 40, 60, 80, 100].map(tick => `
      <line x1="-5" y1="${height - (tick / 100) * height}" x2="0" y2="${height - (tick / 100) * height}" stroke="#333"/>
      <text x="-10" y="${height - (tick / 100) * height + 4}" class="y-tick-label">${tick}%</text>
    `).join('')}
    
    <!-- Axis titles -->
    <text transform="rotate(-90)" x="${-height / 2}" y="-35" class="axis-label">Survival Rate</text>
    <text x="${width / 2}" y="${height + 55}" class="axis-label">Gender</text>
  </g>
</svg>`;

    fs.writeFileSync(path.join(__dirname, '../gender-survival.svg'), svg);
  }

  private createInteractiveHTML(): void {
    console.log('🌐 CREATING INTERACTIVE HTML DASHBOARD');
    console.log('=====================================');

    const html = `<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Titanic Dataset Visualization</title>
    <style>
        body {
            font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
            margin: 0;
            padding: 20px;
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            min-height: 100vh;
        }
        .container {
            max-width: 1200px;
            margin: 0 auto;
            background: white;
            border-radius: 15px;
            box-shadow: 0 20px 40px rgba(0,0,0,0.1);
            overflow: hidden;
        }
        .header {
            background: linear-gradient(135deg, #2c3e50 0%, #3498db 100%);
            color: white;
            padding: 40px;
            text-align: center;
        }
        .header h1 {
            margin: 0;
            font-size: 2.5em;
            font-weight: 300;
        }
        .header p {
            margin: 10px 0 0 0;
            opacity: 0.9;
            font-size: 1.1em;
        }
        .stats-grid {
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(250px, 1fr));
            gap: 20px;
            padding: 40px;
            background: #f8f9fa;
        }
        .stat-card {
            background: white;
            padding: 25px;
            border-radius: 10px;
            box-shadow: 0 5px 15px rgba(0,0,0,0.08);
            text-align: center;
            transition: transform 0.3s ease;
        }
        .stat-card:hover {
            transform: translateY(-5px);
        }
        .stat-number {
            font-size: 2.5em;
            font-weight: bold;
            color: #3498db;
            margin: 0;
        }
        .stat-label {
            color: #666;
            margin: 5px 0 0 0;
            font-size: 1.1em;
        }
        .charts-section {
            padding: 40px;
        }
        .charts-grid {
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(300px, 1fr));
            gap: 30px;
            margin-top: 30px;
        }
        .chart-container {
            background: white;
            border-radius: 10px;
            box-shadow: 0 5px 15px rgba(0,0,0,0.08);
            padding: 20px;
            text-align: center;
        }
        .chart-container svg {
            max-width: 100%;
            height: auto;
        }
        .insights {
            background: #2c3e50;
            color: white;
            padding: 40px;
        }
        .insights h2 {
            color: #3498db;
            margin-top: 0;
        }
        .insight-item {
            margin: 15px 0;
            padding: 15px;
            background: rgba(52, 152, 219, 0.1);
            border-radius: 8px;
            border-left: 4px solid #3498db;
        }
        .footer {
            text-align: center;
            padding: 20px;
            background: #34495e;
            color: white;
        }
    </style>
</head>
<body>
    <div class="container">
        <div class="header">
            <h1>🚢 Titanic Dataset Analysis</h1>
            <p>Interactive visualization of the famous Titanic passenger dataset</p>
        </div>
        
        <div class="stats-grid">
            <div class="stat-card">
                <div class="stat-number">${this.data.length}</div>
                <div class="stat-label">Total Passengers</div>
            </div>
            <div class="stat-card">
                <div class="stat-number">${this.data.filter(p => p.Survived === 1).length}</div>
                <div class="stat-label">Survivors</div>
            </div>
            <div class="stat-card">
                <div class="stat-number">${(this.data.filter(p => p.Survived === 1).length / this.data.length * 100).toFixed(1)}%</div>
                <div class="stat-label">Survival Rate</div>
            </div>
            <div class="stat-card">
                <div class="stat-number">${this.median(this.data.filter(p => p.Age !== null).map(p => p.Age!)).toFixed(0)}</div>
                <div class="stat-label">Median Age</div>
            </div>
        </div>
        
        <div class="charts-section">
            <h2 style="text-align: center; color: #2c3e50; margin-bottom: 10px;">Data Visualizations</h2>
            <p style="text-align: center; color: #666; margin-bottom: 30px;">Explore the patterns and insights from the Titanic dataset</p>
            
            <div class="charts-grid">
                <div class="chart-container">
                    <object data="./survival-by-class.svg" type="image/svg+xml" width="100%"></object>
                </div>
                <div class="chart-container">
                    <object data="./gender-survival.svg" type="image/svg+xml" width="100%"></object>
                </div>
                <div class="chart-container">
                    <object data="./age-distribution.svg" type="image/svg+xml" width="100%"></object>
                </div>
                <div class="chart-container">
                    <object data="./fare-distribution.svg" type="image/svg+xml" width="100%"></object>
                </div>
            </div>
        </div>
        
        <div class="insights">
            <h2>📊 Key Insights</h2>
            <div class="insight-item">
                <strong>Class Matters:</strong> First-class passengers had a ${((this.data.filter(p => p.Pclass === 1 && p.Survived === 1).length / this.data.filter(p => p.Pclass === 1).length) * 100).toFixed(1)}% survival rate, while third-class had only ${((this.data.filter(p => p.Pclass === 3 && p.Survived === 1).length / this.data.filter(p => p.Pclass === 3).length) * 100).toFixed(1)}%.
            </div>
            <div class="insight-item">
                <strong>Gender Disparity:</strong> Women had a ${((this.data.filter(p => p.Sex === 'female' && p.Survived === 1).length / this.data.filter(p => p.Sex === 'female').length) * 100).toFixed(1)}% survival rate compared to ${((this.data.filter(p => p.Sex === 'male' && p.Survived === 1).length / this.data.filter(p => p.Sex === 'male').length) * 100).toFixed(1)}% for men.
            </div>
            <div class="insight-item">
                <strong>Age Factor:</strong> The median age of passengers was ${this.median(this.data.filter(p => p.Age !== null).map(p => p.Age!)).toFixed(0)} years, with ${this.data.filter(p => p.Age === null).length} passengers having missing age data.
            </div>
            <div class="insight-item">
                <strong>Economic Impact:</strong> Fare distribution shows clear class segregation, with luxury passengers paying significantly more for their tickets.
            </div>
        </div>
        
        <div class="footer">
            <p>Generated with TypeScript • Data visualization powered by SVG</p>
            <p style="margin: 5px 0 0 0; opacity: 0.7;">Dataset: Titanic passenger manifest from Kaggle</p>
        </div>
    </div>
</body>
</html>`;

    fs.writeFileSync(path.join(__dirname, '../titanic-dashboard.html'), html);
    console.log('Interactive HTML dashboard created: titanic-dashboard.html\n');
  }

  // Helper methods for statistical calculations
  private mean(values: number[]): number {
    return values.reduce((sum, val) => sum + val, 0) / values.length;
  }

  private median(values: number[]): number {
    const sorted = [...values].sort((a, b) => a - b);
    const mid = Math.floor(sorted.length / 2);
    return sorted.length % 2 !== 0 ? sorted[mid] : (sorted[mid - 1] + sorted[mid]) / 2;
  }

  private createBins(values: number[], min: number, max: number, binCount: number): Array<{min: number, max: number, count: number}> {
    const binSize = (max - min) / binCount;
    const bins = [];
    
    for (let i = 0; i < binCount; i++) {
      const binMin = min + i * binSize;
      const binMax = min + (i + 1) * binSize;
      const count = values.filter(v => v >= binMin && v < binMax).length;
      bins.push({ min: binMin, max: binMax, count });
    }
    
    return bins;
  }
}

// Run the visualization
console.log('🚢 Starting Titanic Data Visualization...\n');
new TitanicVisualizer();