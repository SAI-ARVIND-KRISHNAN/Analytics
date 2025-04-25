import pandas as pd
import os

def parse_duration(text):
    """Convert '1h 2m 30s' to total minutes as float"""
    if pd.isna(text):
        return None
    try:
        parts = text.lower().replace('h', 'h ').replace('m', 'm ').replace('s', 's ').split()
        total_seconds = 0
        for part in parts:
            if 'h' in part:
                total_seconds += int(part.replace('h', '')) * 3600
            elif 'm' in part:
                total_seconds += int(part.replace('m', '')) * 60
            elif 's' in part:
                total_seconds += int(part.replace('s', ''))
        return total_seconds / 60  # return minutes
    except:
        return None

def load_data(directory):
    all_files = [f for f in os.listdir(directory) if f.endswith(".xls")]
    combined_df = pd.DataFrame()

    for file in all_files:
        file_path = os.path.join(directory, file)
        print(f"Reading file: {file_path}")

        try:
            xls_sheets = pd.read_excel(file_path, sheet_name=None)
        except Exception as e:
            print(f"Failed to read {file_path}: {e}")
            continue

        sheet_names = list(xls_sheets.keys())
        if not sheet_names:
            continue

        first_sheet = xls_sheets[sheet_names[0]]

        try:
            df = first_sheet.iloc[1:-2]  # Remove headers and total row
            df = df.rename(columns={df.columns[0]: 'app_name', df.columns[-1]: 'total_time'})
            df = df[['app_name', 'total_time']]
            df['total_time'] = df['total_time'].apply(parse_duration)
            df.dropna(subset=['total_time'], inplace=True)
            df['category'] = 'uncategorized'
            df['user_id'] = os.path.splitext(file)[0]
            combined_df = pd.concat([combined_df, df], ignore_index=True)
        except Exception as e:
            print(f"Skipped {file_path}: structure mismatch → {e}")

    return combined_df
