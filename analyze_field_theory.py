#!/usr/bin/env python3
"""
Comprehensive analysis script for field theory codebase.
Analyzes functions, classes, usage patterns, and physics implementation.
"""

import ast
import os
import re
from collections import defaultdict
from pathlib import Path


def extract_functions_and_classes(file_path):
    """Extract all function and class definitions from a Python file."""
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            content = f.read()
        
        tree = ast.parse(content)
        functions = []
        classes = []
        
        for node in ast.walk(tree):
            if isinstance(node, ast.FunctionDef):
                functions.append({
                    'name': node.name,
                    'line': node.lineno,
                    'args': [arg.arg for arg in node.args.args],
                    'is_method': False  # We'll determine this later
                })
            elif isinstance(node, ast.ClassDef):
                classes.append({
                    'name': node.name,
                    'line': node.lineno,
                    'methods': []
                })
                # Extract methods
                for item in node.body:
                    if isinstance(item, ast.FunctionDef):
                        classes[-1]['methods'].append({
                            'name': item.name,
                            'line': item.lineno,
                            'args': [arg.arg for arg in item.args.args]
                        })
        
        return functions, classes
    except Exception as e:
        print(f"Error parsing {file_path}: {e}")
        return [], []


def find_function_usage(function_name, search_paths):
    """Find where a function is called across the codebase."""
    usage_sites = []
    
    for search_path in search_paths:
        for py_file in Path(search_path).rglob("*.py"):
            try:
                with open(py_file, 'r', encoding='utf-8') as f:
                    content = f.read()
                
                # Look for function calls (simple regex)
                pattern = rf'\b{re.escape(function_name)}\s*\('
                matches = re.finditer(pattern, content)
                
                for match in matches:
                    line_num = content[:match.start()].count('\n') + 1
                    usage_sites.append({
                        'file': str(py_file),
                        'line': line_num
                    })
            except Exception:
                continue
    
    return usage_sites


def analyze_physics_implementation(file_path):
    """Analyze physics-related implementation issues."""
    issues = []
    
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            lines = f.readlines()
        
        for i, line in enumerate(lines, 1):
            # Look for physics-related issues
            if 'TODO' in line or 'FIXME' in line:
                issues.append(f"Line {i}: Incomplete implementation - {line.strip()}")
            
            if 'placeholder' in line.lower():
                issues.append(f"Line {i}: Placeholder code - {line.strip()}")
            
            if 'except Exception:' in line and 'pass' in lines[min(i, len(lines)-1)]:
                issues.append(f"Line {i}: Silent exception handling - {line.strip()}")
            
            if 'raise NotImplementedError' in line:
                issues.append(f"Line {i}: Not implemented - {line.strip()}")
                
    except Exception:
        pass
    
    return issues


def main():
    """Main analysis function."""
    field_theory_path = "/home/feynman/projects/relativistic_turbulence_rg/rtrg/field_theory"
    project_root = "/home/feynman/projects/relativistic_turbulence_rg"
    
    # Search paths for usage analysis
    search_paths = [
        f"{project_root}/rtrg",
        f"{project_root}/tests",
        f"{project_root}/src"
    ]
    
    analysis_results = {
        'files': {},
        'total_functions': 0,
        'total_classes': 0,
        'unused_functions': [],
        'physics_issues': {},
        'usage_stats': defaultdict(int)
    }
    
    # Analyze each file in field theory
    for py_file in Path(field_theory_path).glob("*.py"):
        if py_file.name == '__init__.py':
            continue
            
        print(f"Analyzing {py_file.name}...")
        
        functions, classes = extract_functions_and_classes(py_file)
        physics_issues = analyze_physics_implementation(py_file)
        
        analysis_results['files'][py_file.name] = {
            'functions': functions,
            'classes': classes,
            'physics_issues': physics_issues,
            'line_count': len(open(py_file).readlines())
        }
        
        analysis_results['total_functions'] += len(functions)
        analysis_results['total_classes'] += len(classes)
        analysis_results['physics_issues'][py_file.name] = physics_issues
        
        # Analyze function usage
        for func in functions:
            usage = find_function_usage(func['name'], search_paths)
            func['usage_sites'] = usage
            analysis_results['usage_stats'][len(usage)] += 1
            
            if len(usage) == 0:
                analysis_results['unused_functions'].append({
                    'file': py_file.name,
                    'function': func['name'],
                    'line': func['line']
                })
        
        # Analyze class usage
        for cls in classes:
            usage = find_function_usage(cls['name'], search_paths)
            cls['usage_sites'] = usage
            
            if len(usage) == 0:
                analysis_results['unused_functions'].append({
                    'file': py_file.name,
                    'function': cls['name'] + ' (class)',
                    'line': cls['line']
                })
    
    return analysis_results


if __name__ == "__main__":
    results = main()
    
    print("\n=== ANALYSIS SUMMARY ===")
    print(f"Total functions: {results['total_functions']}")
    print(f"Total classes: {results['total_classes']}")
    print(f"Unused functions/classes: {len(results['unused_functions'])}")
    print(f"Files with physics issues: {sum(1 for issues in results['physics_issues'].values() if issues)}")
    
    # Save results to pickle for the report generator
    import pickle
    with open("/tmp/field_theory_analysis.pkl", "wb") as f:
        pickle.dump(results, f)
    
    print("\nResults saved to /tmp/field_theory_analysis.pkl")