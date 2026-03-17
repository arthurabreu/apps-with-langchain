"""
Implementation of file exporters for the LangChain application.
"""

import json
import os
import pandas as pd
from typing import Dict, Any, List
from src.core.interfaces import IFileExporter
from src.core.exceptions import ModelValidationError


class ExcelExporter(IFileExporter):
    """
    Exporter that converts MD, TXT, or JSON files to Excel sheets.
    Uses pandas and openpyxl for Excel generation.
    """

    def export_to_excel(self, input_file: str, output_file: str) -> bool:
        """
        Export contents of an input file to an Excel sheet.
        
        Args:
            input_file: Path to the source file (.md, .txt, or .json).
            output_file: Path to the target Excel file (.xlsx).
            
        Returns:
            True if successful, False otherwise.
        """
        if not os.path.exists(input_file):
            raise ModelValidationError(f"Input file not found: {input_file}")

        file_ext = os.path.splitext(input_file)[1].lower()
        
        try:
            if file_ext == '.json':
                data = self._read_json(input_file)
            elif file_ext in ['.md', '.txt']:
                data = self._read_text(input_file)
            else:
                raise ModelValidationError(f"Unsupported file format: {file_ext}")

            # Convert data to DataFrame
            df = self._to_dataframe(data)
            
            # Export to Excel
            df.to_excel(output_file, index=False, engine='openpyxl')
            return True
            
        except Exception as e:
            # In a real app, we might want to log this via LoggingService
            print(f"Error during export: {str(e)}")
            return False

    def _read_json(self, file_path: str) -> Any:
        """Read JSON file content, with fallback for empty or invalid JSON."""
        with open(file_path, 'r', encoding='utf-8') as f:
            content = f.read().strip()

        if not content:
            raise ValueError("Model returned an empty response. The model may not support instruction-following (try an instruct/chat variant of the model).")

        try:
            return json.loads(content)
        except json.JSONDecodeError:
            # Response is not JSON — treat as plain text so export still works
            return content

    def _read_text(self, file_path: str) -> str:
        """Read text or markdown file content."""
        with open(file_path, 'r', encoding='utf-8') as f:
            return f.read()

    def _to_dataframe(self, data: Any) -> pd.DataFrame:
        """
        Convert various data formats to a pandas DataFrame.
        
        Special handling for Markdown tables and structured text sections.
        If it's a string (MD/TXT), it tries to find tables or split by sections.
        If it's a list of dicts (JSON), it creates columns for each key.
        If it's a simple dict, it creates rows from items.
        """
        if isinstance(data, str):
            # 1. Try to find a Markdown table
            if '|' in data and '-' in data:
                try:
                    # Very simple MD table extractor
                    # Search for the first table-like structure
                    lines = data.split('\n')
                    table_start = -1
                    for i, line in enumerate(lines):
                        if '|' in line and i + 1 < len(lines) and '-' in lines[i+1] and '|' in lines[i+1]:
                            table_start = i
                            break
                    
                    if table_start != -1:
                        # Extract table lines
                        table_lines = []
                        for i in range(table_start, len(lines)):
                            if '|' in lines[i]:
                                table_lines.append(lines[i])
                            else:
                                break
                        
                        if len(table_lines) >= 3:
                            # Parse MD table using pandas
                            # Remove leading/trailing pipes and whitespace
                            clean_lines = []
                            for line in table_lines:
                                # Remove first and last pipe if they exist
                                l = line.strip()
                                if l.startswith('|'): l = l[1:]
                                if l.endswith('|'): l = l[:-1]
                                clean_lines.append(l)
                            
                            # Skip the header separator line (e.g., |---|---|)
                            header = [c.strip() for c in clean_lines[0].split('|')]
                            rows = []
                            for i in range(2, len(clean_lines)):
                                row = [c.strip() for c in clean_lines[i].split('|')]
                                # Pad or truncate to match header length
                                if len(row) < len(header):
                                    row.extend([''] * (len(header) - len(row)))
                                else:
                                    row = row[:len(header)]
                                rows.append(row)
                            
                            return pd.DataFrame(rows, columns=header)
                except Exception as e:
                    print(f"[DEBUG] Markdown table parsing failed: {e}. Falling back to text structure.")

            # 2. Try to split by sections (headers)
            # If there are headers like ## Section or 1. Section
            import re
            sections = re.split(r'\n(?=[#]{1,3}\s|[0-9]\.\s|[A-Z][a-z]+\s[A-Z][a-z]+:)', data)
            if len(sections) > 1:
                structured_data = []
                for section in sections:
                    lines = section.strip().split('\n', 1)
                    if len(lines) == 2:
                        header = lines[0].strip().replace('#', '').strip()
                        content = lines[1].strip()
                        structured_data.append({"Section": header, "Details": content})
                    else:
                        structured_data.append({"Content": section.strip()})
                return pd.DataFrame(structured_data)

            # 3. Fallback: single cell
            return pd.DataFrame([{"Content": data}])
        
        if isinstance(data, list):
            # List of objects (typical JSON array)
            return pd.DataFrame(data)
        
        if isinstance(data, dict):
            # Check if it's a simple dict or nested
            # If all values are simple types, we can make it a single row
            if all(not isinstance(v, (dict, list)) for v in data.values()):
                return pd.DataFrame([data])
            else:
                # Otherwise, represent as key-value pairs
                return pd.DataFrame(list(data.items()), columns=["Key", "Value"])
        
        # Fallback
        return pd.DataFrame([{"Data": str(data)}])
