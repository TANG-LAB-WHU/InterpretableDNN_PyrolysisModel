"""
Missing value handling utility for municipal sludge data.

This script offers several strategies to process missing data:
1. Drop rows/columns that contain missing values
2. Fill with mean / median / mode / constant
3. Forward fill / backward fill
4. Interpolation (linear / polynomial / spline)
5. Machine-learning based prediction (KNN, MICE)

All interface messages, doc-strings and labels are in English.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.impute import SimpleImputer, KNNImputer
from sklearn.experimental import enable_iterative_imputer  # noqa: F401  pylint: disable=unused-import
from sklearn.impute import IterativeImputer
from sklearn.ensemble import RandomForestRegressor
import warnings
warnings.filterwarnings('ignore')

# Configure fonts (remove Chinese-specific settings)
# Adjust if you need specific fonts locally.
plt.rcParams['axes.unicode_minus'] = False

class MissingValueHandler:
    """Missing value handling class"""
    
    def __init__(self, filepath):
        """
        Initialize the missing value processor
        
        Args:
            filepath: Excel file path
        """
        self.filepath = filepath
        self.data = None
        self.original_data = None
        
    def load_data(self):
        """Load data"""
        try:
            print(f"Loading data file: {self.filepath}")
            self.data = pd.read_excel(self.filepath)
            self.original_data = self.data.copy()
            print(f"Data loaded successfully! Data shape: {self.data.shape}")
            return True
        except Exception as e:
            print(f"Data loading failed: {e}")
            return False
    
    def analyze_missing_values(self):
        """Analyze missing value situation"""
        if self.data is None:
            print("Please load data first!")
            return
        
        print("\n" + "="*50)
        print("Missing value analysis report")
        print("="*50)
        
        # Basic information
        print(f"Data set shape: {self.data.shape}")
        print(f"Total data points: {self.data.size}")
        
        # Missing value statistics
        missing_count = self.data.isnull().sum()
        missing_percent = (missing_count / len(self.data)) * 100
        
        missing_summary = pd.DataFrame({
            'Column Name': self.data.columns,
            'Missing Count': missing_count.values,
            'Missing Percentage': missing_percent.values
        })
        missing_summary = missing_summary[missing_summary['Missing Count'] > 0]
        missing_summary = missing_summary.sort_values('Missing Count', ascending=False)
        
        if len(missing_summary) > 0:
            print(f"\nFound {len(missing_summary)} columns with missing values:")
            print(missing_summary.to_string(index=False))
            
            # Missing value visualization
            self.visualize_missing_values()
        else:
            print("\nCongratulations! No missing values found in the dataset.")
        
        return missing_summary
    
    def visualize_missing_values(self):
        """Visualize missing value distribution"""
        # Create missing value heatmap
        plt.figure(figsize=(12, 8))
        
        plt.subplot(2, 2, 1)
        sns.heatmap(self.data.isnull(), cbar=True, cmap='viridis', 
                   yticklabels=False, xticklabels=True)
        plt.title('Missing value distribution heatmap')
        plt.xticks(rotation=45)
        
        # Missing value statistics bar chart
        plt.subplot(2, 2, 2)
        missing_count = self.data.isnull().sum()
        missing_cols = missing_count[missing_count > 0]
        if len(missing_cols) > 0:
            missing_cols.plot(kind='bar')
            plt.title('Missing value count per column')
            plt.xticks(rotation=45)
            plt.ylabel('Missing value count')
        
        # Missing value percentage
        plt.subplot(2, 2, 3)
        if len(missing_cols) > 0:
            missing_percent = (missing_cols / len(self.data)) * 100
            missing_percent.plot(kind='bar', color='orange')
            plt.title('Missing value percentage per column')
            plt.xticks(rotation=45)
            plt.ylabel('Missing value percentage (%)')
        
        # Missing value pattern
        plt.subplot(2, 2, 4)
        missing_matrix = self.data.isnull().astype(int)
        if missing_matrix.sum().sum() > 0:
            plt.imshow(missing_matrix.T, cmap='viridis', aspect='auto')
            plt.title('Missing value pattern')
            plt.xlabel('Sample index')
            plt.ylabel('Feature')
        
        plt.tight_layout()
        plt.savefig('missing_values_analysis.png', dpi=300, bbox_inches='tight')
        plt.show()
    
    def remove_missing_values(self, axis=0, threshold=0.5):
        """
        Drop rows or columns that contain missing values
        
        Args:
            axis: 0 drop rows, 1 drop columns
            threshold: Missing value ratio threshold (0-1)
        """
        if self.data is None:
            print("Please load data first!")
            return None
        
        original_shape = self.data.shape
        
        if axis == 0:  # Drop rows
            # Calculate missing value ratio per row
            missing_ratio = self.data.isnull().sum(axis=1) / self.data.shape[1]
            # Keep rows with missing value ratio below threshold
            cleaned_data = self.data[missing_ratio <= threshold].copy()
            print(f"Dropped rows with missing value ratio above {threshold*100}%")
        else:  # Drop columns
            # Calculate missing value ratio per column
            missing_ratio = self.data.isnull().sum() / self.data.shape[0]
            # Keep columns with missing value ratio below threshold
            cols_to_keep = missing_ratio[missing_ratio <= threshold].index
            cleaned_data = self.data[cols_to_keep].copy()
            print(f"Dropped columns with missing value ratio above {threshold*100}%")
        
        print(f"Original data shape: {original_shape}")
        print(f"Cleaned data shape: {cleaned_data.shape}")
        print(f"Dropped {original_shape[0] - cleaned_data.shape[0]} rows, {original_shape[1] - cleaned_data.shape[1]} columns")
        
        return cleaned_data
    
    def fill_missing_simple(self, strategy='mean', columns=None):
        """
        Simple fill missing values
        
        Args:
            strategy: 'mean', 'median', 'mode', 'constant'
            columns: Specify columns, None for all numeric columns
        """
        if self.data is None:
            print("Please load data first!")
            return None
        
        filled_data = self.data.copy()
        
        if columns is None:
            # Get numeric columns
            numeric_columns = filled_data.select_dtypes(include=[np.number]).columns
        else:
            numeric_columns = columns
        
        print(f"Filling missing values using {strategy} strategy...")
        
        for col in numeric_columns:
            if filled_data[col].isnull().sum() > 0:
                if strategy == 'mean':
                    fill_value = filled_data[col].mean()
                elif strategy == 'median':
                    fill_value = filled_data[col].median()
                elif strategy == 'mode':
                    fill_value = filled_data[col].mode()[0] if not filled_data[col].mode().empty else 0
                elif strategy == 'constant':
                    fill_value = 0
                else:
                    print(f"Unknown strategy: {strategy}")
                    continue
                
                filled_data[col].fillna(fill_value, inplace=True)
                print(f"Column '{col}': Filled {self.data[col].isnull().sum()} missing values with {fill_value:.3f}")
        
        return filled_data
    
    def fill_missing_interpolation(self, method='linear'):
        """
        Fill missing values using interpolation
        
        Args:
            method: 'linear', 'polynomial', 'spline'
        """
        if self.data is None:
            print("Please load data first!")
            return None
        
        filled_data = self.data.copy()
        numeric_columns = filled_data.select_dtypes(include=[np.number]).columns
        
        print(f"Filling missing values using {method} interpolation...")
        
        for col in numeric_columns:
            if filled_data[col].isnull().sum() > 0:
                missing_count = filled_data[col].isnull().sum()
                if method == 'linear':
                    filled_data[col] = filled_data[col].interpolate(method='linear')
                elif method == 'polynomial':
                    filled_data[col] = filled_data[col].interpolate(method='polynomial', order=2)
                elif method == 'spline':
                    filled_data[col] = filled_data[col].interpolate(method='spline', order=1)
                
                print(f"Column '{col}': Interpolated {missing_count} missing values")
        
        return filled_data
    
    def fill_missing_knn(self, n_neighbors=5):
        """
        Fill missing values using KNN
        
        Args:
            n_neighbors: Number of K-nearest neighbors
        """
        if self.data is None:
            print("Please load data first!")
            return None
        
        filled_data = self.data.copy()
        numeric_columns = filled_data.select_dtypes(include=[np.number]).columns
        
        print(f"Filling missing values using KNN algorithm (K={n_neighbors})...")
        
        if len(numeric_columns) > 0:
            imputer = KNNImputer(n_neighbors=n_neighbors)
            filled_data[numeric_columns] = imputer.fit_transform(filled_data[numeric_columns])
            
            for col in numeric_columns:
                missing_count = self.data[col].isnull().sum()
                if missing_count > 0:
                    print(f"Column '{col}': KNN filled {missing_count} missing values")
        
        return filled_data
    
    def fill_missing_iterative(self, max_iter=10):
        """
        Fill missing values using iterative algorithm (MICE algorithm)
        
        Args:
            max_iter: Maximum number of iterations
        """
        if self.data is None:
            print("Please load data first!")
            return None
        
        filled_data = self.data.copy()
        numeric_columns = filled_data.select_dtypes(include=[np.number]).columns
        
        print(f"Filling missing values using iterative algorithm (MICE)...")
        
        if len(numeric_columns) > 0:
            imputer = IterativeImputer(
                estimator=RandomForestRegressor(n_estimators=10, random_state=42),
                max_iter=max_iter,
                random_state=42
            )
            filled_data[numeric_columns] = imputer.fit_transform(filled_data[numeric_columns])
            
            for col in numeric_columns:
                missing_count = self.data[col].isnull().sum()
                if missing_count > 0:
                    print(f"Column '{col}': Iterative filled {missing_count} missing values")
        
        return filled_data
    
    def fill_fixed_carbon_with_constraints(self):
        """
        Fill Fixed Carbon missing values based on chemical constraints
        Constraint: Volatile Matter + Fixed Carbon + Ash ≈ 100%
        """
        if self.data is None:
            print("Please load data first!")
            return None
        
        filled_data = self.data.copy()
        
        # Find related column names (support different naming methods)
        vm_cols = [col for col in filled_data.columns if 'volatile' in col.lower() or 'vm' in col.lower()]
        fc_cols = [col for col in filled_data.columns if 'fixed' in col.lower() and 'carbon' in col.lower()]
        ash_cols = [col for col in filled_data.columns if 'ash' in col.lower() and '/' in col]
        
        vm_col = vm_cols[0] if vm_cols else None
        fc_col = fc_cols[0] if fc_cols else None  
        ash_col = ash_cols[0] if ash_cols else None
        
        if not all([vm_col, fc_col, ash_col]):
            print("Cannot find Volatile Matter, Fixed Carbon, or Ash columns")
            return filled_data
        
        print(f"Applying chemical constraint fill for Fixed Carbon...")
        print(f"Using columns: {vm_col}, {fc_col}, {ash_col}")
        
        filled_count = 0
        
        for idx, row in filled_data.iterrows():
            # If Fixed Carbon is missing but VM and Ash are known
            if pd.isna(row[fc_col]) and pd.notna(row[vm_col]) and pd.notna(row[ash_col]):
                calculated_fc = 100 - row[vm_col] - row[ash_col]
                
                # Verify the reasonability of the calculation result
                if 0 <= calculated_fc <= 100:
                    filled_data.loc[idx, fc_col] = calculated_fc
                    filled_count += 1
                    print(f"Row {idx}: Fixed Carbon = 100 - {row[vm_col]:.1f} - {row[ash_col]:.1f} = {calculated_fc:.1f}%")
        
        print(f"Based on chemical constraints, {filled_count} Fixed Carbon missing values were filled")
        return filled_data
    
    def fill_proximate_analysis_constraints(self):
        """
        Fill missing values based on industrial analysis constraints
        """
        if self.data is None:
            print("Please load data first!")
            return None
        
        filled_data = self.data.copy()
        
        # Find industrial analysis related columns
        vm_cols = [col for col in filled_data.columns if 'volatile' in col.lower()]
        fc_cols = [col for col in filled_data.columns if 'fixed' in col.lower() and 'carbon' in col.lower()]
        ash_cols = [col for col in filled_data.columns if 'ash' in col.lower() and '/' in col]
        moisture_cols = [col for col in filled_data.columns if 'moisture' in col.lower() or 'water' in col.lower()]
        
        vm_col = vm_cols[0] if vm_cols else None
        fc_col = fc_cols[0] if fc_cols else None
        ash_col = ash_cols[0] if ash_cols else None
        moisture_col = moisture_cols[0] if moisture_cols else None
        
        components = [vm_col, fc_col, ash_col]
        if moisture_col:
            components.append(moisture_col)
        
        # Remove None values
        components = [comp for comp in components if comp is not None]
        
        if len(components) < 3:
            print("Insufficient industrial analysis components to apply constraint fill")
            return filled_data
        
        print(f"Applying industrial analysis constraint fill: {components}")
        
        filled_count = 0
        
        for idx, row in filled_data.iterrows():
            # Calculate the sum of known components
            known_sum = 0
            missing_components = []
            
            for comp in components:
                if pd.notna(row[comp]):
                    known_sum += row[comp]
                else:
                    missing_components.append(comp)
            
            # If only one component is missing, it can be calculated based on constraints
            if len(missing_components) == 1:
                missing_comp = missing_components[0]
                calculated_value = 100 - known_sum
                
                # Verify the reasonability of the calculation result
                if 0 <= calculated_value <= 100:
                    filled_data.loc[idx, missing_comp] = calculated_value
                    filled_count += 1
                    print(f"Row {idx}: {missing_comp} = 100 - {known_sum:.1f} = {calculated_value:.1f}%")
        
        print(f"Based on industrial analysis constraints, a total of {filled_count} missing values were filled")
        return filled_data
    
    def compare_methods(self, methods=['mean', 'median', 'knn', 'iterative', 'constraints']):
        """
        Compare different fill methods
        
        Args:
            methods: List of methods to compare
        """
        if self.data is None:
            print("Please load data first!")
            return
        
        results = {}
        
        print("\nComparing different fill methods...")
        
        for method in methods:
            print(f"\nTesting method: {method}")
            
            if method == 'mean':
                filled_data = self.fill_missing_simple(strategy='mean')
            elif method == 'median':
                filled_data = self.fill_missing_simple(strategy='median')
            elif method == 'mode':
                filled_data = self.fill_missing_simple(strategy='mode')
            elif method == 'knn':
                filled_data = self.fill_missing_knn(n_neighbors=5)
            elif method == 'iterative':
                filled_data = self.fill_missing_iterative(max_iter=10)
            elif method == 'interpolation':
                filled_data = self.fill_missing_interpolation(method='linear')
            elif method == 'constraints':
                # First apply chemical constraints, then use KNN for remaining missing values
                filled_data = self.fill_proximate_analysis_constraints()
                if filled_data.isnull().sum().sum() > 0:
                    filled_data = self.fill_missing_knn_on_data(filled_data, n_neighbors=5)
            else:
                print(f"Unknown method: {method}")
                continue
            
            # Calculate statistics after filling
            remaining_missing = filled_data.isnull().sum().sum()
            results[method] = {
                'data': filled_data,
                'remaining_missing': remaining_missing
            }
            
            print(f"Remaining missing values: {remaining_missing}")
        
        return results
    
    def fill_missing_knn_on_data(self, data, n_neighbors=5):
        """
        Perform KNN fill on specified data
        """
        filled_data = data.copy()
        numeric_columns = filled_data.select_dtypes(include=[np.number]).columns
        
        if len(numeric_columns) > 0:
            imputer = KNNImputer(n_neighbors=n_neighbors)
            filled_data[numeric_columns] = imputer.fit_transform(filled_data[numeric_columns])
        
        return filled_data
    
    def save_cleaned_data(self, data, filename_suffix='_cleaned'):
        """
        Save cleaned data
        
        Args:
            data: Cleaned data
            filename_suffix: File name suffix
        """
        if data is None:
            print("No data to save!")
            return
        
        # Generate output file name
        base_name = self.filepath.replace('.xlsx', '').replace('.xls', '')
        output_filename = f"{base_name}{filename_suffix}.xlsx"
        
        try:
            data.to_excel(output_filename, index=False)
            print(f"Cleaned data saved to: {output_filename}")
            
            # Save cleaning report
            report_filename = f"{base_name}{filename_suffix}_report.txt"
            with open(report_filename, 'w', encoding='utf-8') as f:
                f.write("Data cleaning report\n")
                f.write("="*50 + "\n")
                f.write(f"Original data shape: {self.original_data.shape}\n")
                f.write(f"Cleaned data shape: {data.shape}\n")
                f.write(f"Original missing value total: {self.original_data.isnull().sum().sum()}\n")
                f.write(f"Cleaned missing value total: {data.isnull().sum().sum()}\n")
                f.write("\nMissing value situation per column:\n")
                
                for col in data.columns:
                    original_missing = self.original_data[col].isnull().sum() if col in self.original_data.columns else 0
                    current_missing = data[col].isnull().sum()
                    f.write(f"{col}: {original_missing} -> {current_missing}\n")
            
            print(f"Cleaning report saved to: {report_filename}")
            
        except Exception as e:
            print(f"Save failed: {e}")

def main():
    """Main function: Demonstrate missing value processing process"""
    
    # Initialize processor
    handler = MissingValueHandler('Municipal_Sludge_Data.xlsx')
    
    # Load data
    if not handler.load_data():
        return
    
    # Analyze missing values
    missing_summary = handler.analyze_missing_values()
    
    if missing_summary is not None and len(missing_summary) > 0:
        print("\n" + "="*50)
        print("Missing value processing options")
        print("="*50)
        print("1. Drop rows with missing values")
        print("2. Fill with mean")
        print("3. Fill with median")
        print("4. Fill with KNN")
        print("5. Fill with iterative (MICE)")
        print("6. Fill with interpolation")
        print("7. Compare all methods")
        print("8. Custom method combination")
        
        # Here you can add user interaction selection
        # Now default to compare all methods
        print("\nAutomatically running all method comparison...")
        
        # Compare different methods
        results = handler.compare_methods()
        
        # Select the best method (here select the one with the least remaining missing values)
        best_method = min(results.keys(), key=lambda x: results[x]['remaining_missing'])
        best_data = results[best_method]['data']
        
        print(f"\nRecommended method: {best_method}")
        print(f"This method remaining missing values: {results[best_method]['remaining_missing']}")
        
        # Save best result
        handler.save_cleaned_data(best_data, f'_cleaned_{best_method}')
        
        # You can also let the user choose to save multiple results
        for method, result in results.items():
            if result['remaining_missing'] == 0:  # Completely filled method
                handler.save_cleaned_data(result['data'], f'_cleaned_{method}')
    
    print("\nMissing value processing completed!")

if __name__ == "__main__":
    main() 