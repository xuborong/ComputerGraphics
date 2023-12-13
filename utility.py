import pandas as pd


def generate_timecode(num_rows):
    timecodes = []
    for i in range(num_rows):
        # Calculate hours, minutes, seconds, and frames
        total_seconds = i // 60
        frames = i % 60
        hours, remainder = divmod(total_seconds, 3600)
        minutes, seconds = divmod(remainder, 60)

        # Format the timecode
        timecode = f"{int(hours):02}:{int(minutes):02}:{int(seconds):02}:{int(frames):02}.000"
        timecodes.append(timecode)

    return timecodes


def add_timecode_column(df):
    # Generate timecodes based on the number of rows in the DataFrame
    num_rows = len(df)
    if 'Timecode' in df.columns:
        df['Timecode'] = generate_timecode(num_rows)
    else:
        df.insert(0, 'Timecode', generate_timecode(num_rows))
    if 'BlendshapeCount' not in df.columns:
        df.insert(1, 'BlendshapeCount', [61] * num_rows)
    return df
    # Save the modified DataFrame to a new CSV file
    # output_path = csv_path.replace('.csv', '_with_timecodes.csv')
    # df.to_csv(output_path, index=False)
    # print(f"Timecode added and saved to {output_path}")


# Example usage
# csv_path = 'C:/Users/xubor/OneDrive/Desktop/number12.csv'
# file = pd.read_csv(csv_path)
# print(add_timecode_column(file))
