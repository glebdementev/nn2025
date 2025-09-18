#!/usr/bin/env python3
"""
Titanic Survival Factor Analysis
Analyzes the most important factors affecting survival on the Titanic
and generates an SVG visualization for dashboard embedding.
"""

import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from io import StringIO
import re

class TitanicSurvivalAnalysis:
    def __init__(self, csv_path):
        self.df = pd.read_csv(csv_path)
        self.feature_importance = None
        self.features = None
        self.prepare_data()
        
    def prepare_data(self):
        """Clean and prepare data for analysis"""
        print("🔍 Preparing data for survival factor analysis...")
        
        # Create a copy for analysis
        self.analysis_df = self.df.copy()
        
        # Fill missing values
        self.analysis_df['Age'] = self.analysis_df['Age'].fillna(self.analysis_df['Age'].median())
        self.analysis_df['Embarked'] = self.analysis_df['Embarked'].fillna(self.analysis_df['Embarked'].mode()[0])
        self.analysis_df['Fare'] = self.analysis_df['Fare'].fillna(self.analysis_df['Fare'].median())
        
        # Create new engineered features
        self.analysis_df['FamilySize'] = self.analysis_df['SibSp'] + self.analysis_df['Parch'] + 1
        self.analysis_df['IsAlone'] = (self.analysis_df['FamilySize'] == 1).astype(int)
        
        # Age groups
        self.analysis_df['AgeGroup'] = pd.cut(self.analysis_df['Age'], 
                                            bins=[0, 12, 18, 35, 60, 100], 
                                            labels=['Child', 'Teen', 'Adult', 'MiddleAge', 'Senior'])
        
        # Fare groups
        self.analysis_df['FareGroup'] = pd.qcut(self.analysis_df['Fare'], 
                                              q=4, labels=['Low', 'Medium', 'High', 'VeryHigh'])
        
        # Title extraction from names
        self.analysis_df['Title'] = self.analysis_df['Name'].str.extract(r' ([A-Za-z]+)\.', expand=False)
        self.analysis_df['Title'] = self.analysis_df['Title'].replace(['Lady', 'Countess','Capt', 'Col',
                                                                    'Don', 'Dr', 'Major', 'Rev', 'Sir', 'Jonkheer', 'Dona'], 'Rare')
        self.analysis_df['Title'] = self.analysis_df['Title'].replace('Mlle', 'Miss')
        self.analysis_df['Title'] = self.analysis_df['Title'].replace('Ms', 'Miss')
        self.analysis_df['Title'] = self.analysis_df['Title'].replace('Mme', 'Mrs')
        
        print(f"✅ Data prepared: {len(self.analysis_df)} records with engineered features")
        
    def calculate_feature_importance(self):
        """Use Random Forest to calculate feature importance"""
        print("📊 Calculating feature importance using Random Forest...")
        
        # Select features for analysis
        feature_columns = [
            'Pclass', 'Sex', 'Age', 'SibSp', 'Parch', 'Fare', 'Embarked',
            'FamilySize', 'IsAlone', 'AgeGroup', 'FareGroup', 'Title'
        ]
        
        # Prepare features
        X = self.analysis_df[feature_columns].copy()
        y = self.analysis_df['Survived']
        
        # Encode categorical variables
        le = LabelEncoder()
        categorical_columns = ['Sex', 'Embarked', 'AgeGroup', 'FareGroup', 'Title']
        
        for col in categorical_columns:
            X[col] = le.fit_transform(X[col].astype(str))
        
        # Train Random Forest
        rf = RandomForestClassifier(n_estimators=100, random_state=42, max_depth=10)
        rf.fit(X, y)
        
        # Get feature importance
        importance = rf.feature_importances_
        self.features = feature_columns
        self.feature_importance = importance
        
        # Create results dataframe
        importance_df = pd.DataFrame({
            'Feature': feature_columns,
            'Importance': importance
        }).sort_values('Importance', ascending=False)
        
        print("🎯 Top 5 Most Important Survival Factors:")
        for i, row in importance_df.head().iterrows():
            print(f"   {row['Feature']}: {row['Importance']:.3f}")
        
        return importance_df
    
    def calculate_correlation_analysis(self):
        """Calculate correlation between features and survival"""
        print("🔗 Calculating correlation analysis...")
        
        # Select only numerical and encoded features for correlation
        numerical_features = ['Survived', 'Pclass', 'Age', 'SibSp', 'Parch', 'Fare', 'FamilySize', 'IsAlone']
        corr_df = self.analysis_df[numerical_features].copy()
        
        # Encode categorical variables for correlation
        le = LabelEncoder()
        categorical_columns = ['Sex', 'Embarked', 'AgeGroup', 'FareGroup', 'Title']
        
        for col in categorical_columns:
            if col in self.analysis_df.columns:
                corr_df[col] = le.fit_transform(self.analysis_df[col].astype(str))
        
        # Calculate correlation with survival
        survival_corr = corr_df.corr()['Survived'].abs().sort_values(ascending=False)
        
        # Remove survival itself and get top correlations
        survival_corr = survival_corr.drop('Survived')
        
        print("🔗 Top correlations with survival:")
        for feature, corr in survival_corr.head().items():
            print(f"   {feature}: {corr:.3f}")
        
        return survival_corr
    
    def create_survival_rates_by_factor(self):
        """Calculate survival rates for different factor categories"""
        survival_data = {}
        
        # Gender survival rates
        gender_survival = self.analysis_df.groupby('Sex')['Survived'].agg(['count', 'sum', 'mean']).round(3)
        survival_data['Gender'] = {
            'Female': gender_survival.loc['female', 'mean'],
            'Male': gender_survival.loc['male', 'mean']
        }
        
        # Class survival rates  
        class_survival = self.analysis_df.groupby('Pclass')['Survived'].agg(['count', 'sum', 'mean']).round(3)
        survival_data['Class'] = {
            f'Class {cls}': rate for cls, rate in class_survival['mean'].items()
        }
        
        # Age group survival rates
        age_survival = self.analysis_df.groupby('AgeGroup')['Survived'].agg(['count', 'sum', 'mean']).round(3)
        survival_data['Age Group'] = {
            str(age): rate for age, rate in age_survival['mean'].items()
        }
        
        # Family size survival rates
        family_survival = self.analysis_df.groupby('FamilySize')['Survived'].agg(['count', 'sum', 'mean']).round(3)
        survival_data['Family Size'] = {
            f'Size {size}': rate for size, rate in family_survival['mean'].items() if size <= 6
        }
        
        return survival_data
        
    def generate_factor_importance_svg(self):
        """Generate SVG chart showing factor importance"""
        print("🎨 Generating factor importance SVG...")
        
        # Get feature importance
        importance_df = self.calculate_feature_importance()
        
        # Take top 8 factors for clarity
        top_factors = importance_df.head(8)
        
        # Generate SVG
        width, height = 700, 500
        margin = {'top': 60, 'right': 40, 'bottom': 80, 'left': 120}
        chart_width = width - margin['left'] - margin['right']
        chart_height = height - margin['top'] - margin['bottom']
        
        # Colors for bars
        colors = ['#e74c3c', '#3498db', '#2ecc71', '#f39c12', '#9b59b6', '#1abc9c', '#34495e', '#e67e22']
        
        svg_content = f'''<svg width="{width}" height="{height}" xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {width} {height}">
    <defs>
        <style>
            .chart-title {{ font-family: Arial, sans-serif; font-size: 20px; font-weight: bold; text-anchor: middle; fill: #2c3e50; }}
            .axis-label {{ font-family: Arial, sans-serif; font-size: 14px; text-anchor: middle; fill: #34495e; }}
            .bar-label {{ font-family: Arial, sans-serif; font-size: 11px; text-anchor: end; fill: #2c3e50; }}
            .importance-label {{ font-family: Arial, sans-serif; font-size: 10px; text-anchor: start; fill: white; font-weight: bold; }}
            .tick-label {{ font-family: Arial, sans-serif; font-size: 10px; text-anchor: end; fill: #7f8c8d; }}
        </style>
        <linearGradient id="barGradient1" x1="0%" y1="0%" x2="100%" y2="0%">
            <stop offset="0%" style="stop-color:#e74c3c;stop-opacity:0.8" />
            <stop offset="100%" style="stop-color:#c0392b;stop-opacity:1" />
        </linearGradient>
        <linearGradient id="barGradient2" x1="0%" y1="0%" x2="100%" y2="0%">
            <stop offset="0%" style="stop-color:#3498db;stop-opacity:0.8" />
            <stop offset="100%" style="stop-color:#2980b9;stop-opacity:1" />
        </linearGradient>
        <linearGradient id="barGradient3" x1="0%" y1="0%" x2="100%" y2="0%">
            <stop offset="0%" style="stop-color:#2ecc71;stop-opacity:0.8" />
            <stop offset="100%" style="stop-color:#27ae60;stop-opacity:1" />
        </linearGradient>
    </defs>
    
    <g transform="translate({margin['left']},{margin['top']})">
        <!-- Chart title -->
        <text x="{chart_width/2}" y="-20" class="chart-title">Most Important Factors Affecting Titanic Survival</text>
        
        <!-- Bars -->'''
        
        max_importance = top_factors['Importance'].max()
        bar_height = (chart_height - 20) / len(top_factors)
        
        for i, (_, row) in enumerate(top_factors.iterrows()):
            feature = row['Feature']
            importance = row['Importance']
            bar_width = (importance / max_importance) * (chart_width - 100)
            y_pos = i * bar_height + 10
            
            # Clean feature names for display
            display_name = {
                'Sex': 'Gender',
                'Pclass': 'Passenger Class', 
                'Fare': 'Ticket Fare',
                'Age': 'Age',
                'FamilySize': 'Family Size',
                'Title': 'Social Title',
                'Embarked': 'Port of Embarkation',
                'IsAlone': 'Traveling Alone',
                'FareGroup': 'Fare Category',
                'AgeGroup': 'Age Category',
                'SibSp': 'Siblings/Spouses',
                'Parch': 'Parents/Children'
            }.get(feature, feature)
            
            gradient_id = f"barGradient{(i % 3) + 1}"
            
            svg_content += f'''
        <rect x="0" y="{y_pos}" width="{bar_width}" height="{bar_height-5}" 
              fill="url(#{gradient_id})" stroke="#2c3e50" stroke-width="0.5" rx="2"/>
        
        <text x="-5" y="{y_pos + bar_height/2 + 3}" class="bar-label">{display_name}</text>
        
        <text x="{bar_width + 5}" y="{y_pos + bar_height/2 + 3}" class="importance-label" fill="#2c3e50">
            {importance:.3f} ({(importance/max_importance*100):.1f}%)
        </text>'''
        
        # Add axes
        svg_content += f'''
        
        <!-- Y-axis -->
        <line x1="-2" y1="0" x2="-2" y2="{chart_height}" stroke="#bdc3c7" stroke-width="1"/>
        
        <!-- X-axis -->
        <line x1="0" y1="{chart_height}" x2="{chart_width}" y2="{chart_height}" stroke="#bdc3c7" stroke-width="1"/>
        
        <!-- X-axis title -->
        <text x="{chart_width/2}" y="{chart_height + 35}" class="axis-label">Feature Importance Score</text>
        
        <!-- Subtitle -->
        <text x="{chart_width/2}" y="-2" class="axis-label" font-size="12px" fill="#7f8c8d">
            Based on Random Forest Machine Learning Analysis
        </text>
        
    </g>
</svg>'''
        
        return svg_content
    
    def run_complete_analysis(self):
        """Run complete survival factor analysis"""
        print("🚢 Starting Complete Titanic Survival Factor Analysis\n")
        
        # Basic statistics
        print(f"📋 Dataset Overview:")
        print(f"   Total passengers: {len(self.df)}")
        print(f"   Survivors: {self.df['Survived'].sum()}")
        print(f"   Survival rate: {self.df['Survived'].mean():.1%}\n")
        
        # Calculate feature importance
        importance_df = self.calculate_feature_importance()
        print()
        
        # Calculate correlations
        correlations = self.calculate_correlation_analysis()
        print()
        
        # Generate survival rates
        survival_rates = self.create_survival_rates_by_factor()
        
        print("📊 Survival Rates by Key Factors:")
        for factor_type, rates in survival_rates.items():
            print(f"\n   {factor_type}:")
            for category, rate in rates.items():
                print(f"     {category}: {rate:.1%}")
        
        print("\n" + "="*60)
        print("🎯 KEY FINDINGS:")
        print("="*60)
        
        top_factor = importance_df.iloc[0]
        print(f"🥇 MOST IMPORTANT FACTOR: {top_factor['Feature']}")
        print(f"   Importance Score: {top_factor['Importance']:.3f}")
        
        if top_factor['Feature'] == 'Sex':
            female_rate = survival_rates['Gender']['Female']
            male_rate = survival_rates['Gender']['Male']
            print(f"   Female survival: {female_rate:.1%}")
            print(f"   Male survival: {male_rate:.1%}")
            print(f"   Gender gap: {(female_rate - male_rate):.1%}")
        
        print(f"\n🥈 SECOND MOST IMPORTANT: {importance_df.iloc[1]['Feature']}")
        print(f"   Importance Score: {importance_df.iloc[1]['Importance']:.3f}")
        
        print(f"\n🥉 THIRD MOST IMPORTANT: {importance_df.iloc[2]['Feature']}")
        print(f"   Importance Score: {importance_df.iloc[2]['Importance']:.3f}")
        
        return importance_df, survival_rates

def main():
    # Initialize analysis
    analyzer = TitanicSurvivalAnalysis('data/train.csv')
    
    # Run complete analysis
    importance_df, survival_rates = analyzer.run_complete_analysis()
    
    # Generate SVG
    svg_content = analyzer.generate_factor_importance_svg()
    
    # Save SVG file
    with open('survival-factors-importance.svg', 'w') as f:
        f.write(svg_content)
    
    print(f"\n✅ Analysis complete! SVG chart saved as 'survival-factors-importance.svg'")
    print("🎨 Ready to embed in your dashboard!")

if __name__ == "__main__":
    main()
