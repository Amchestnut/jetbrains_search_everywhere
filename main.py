import pandas as pd
import json

# 1) Loading the dataset
file_path = '2024InternshipData.csv'
data = pd.read_csv(file_path)

# 2) Parsing 'event_data' to extract relevant JSON fields
def parse_event_data(row):
    try:
        data = json.loads(row)
        session_id = data.get("session_id", None)
        searchStateFeatures = data.get("searchStateFeatures", {})
        experimentGroup = data.get("experimentGroup", None)
        selectedIndexes = data.get("selectedIndexes", [])
        eventIndex = data.get("eventIndex", None)

        return pd.Series([session_id, searchStateFeatures, experimentGroup, selectedIndexes, eventIndex])
    except json.JSONDecodeError:
        return pd.Series([None, None, None, None, None])

# 3) Apply the parsing function to extract fields from 'event_data'
data[['session_id', 'searchStateFeatures', 'experimentGroup', 'selectedIndexes', 'eventIndex']] = data['event_data'].apply(parse_event_data)

# We dont need this column anymore
data.drop(columns=['event_data'], inplace=True)

# Extract 'queryLength' from 'searchStateFeatures' for easier analysis
# in lambda x, x is searchStateFeatures in each row
# we check if x is a dictionary, which confirms searchStateFeatures was successfully parsed from JSON
# If x is a dictionary, I use x.get('queryLength') to retrieve the queryLength value
data['queryLength'] = data['searchStateFeatures'].apply(lambda x: x.get('queryLength') if isinstance(x, dict) else None)

# Separated data by experiment groups, as requested from the task
group_0 = data[data['experimentGroup'] == 0]
group_1 = data[data['experimentGroup'] == 1]

# Calculating and summarizing metrics for comparison
comparison_metrics = {
    "Total Events": [group_0.shape[0], group_1.shape[0]],
    "Average Query Length": [group_0['queryLength'].mean(), group_1['queryLength'].mean()],
    "Average Selected Index Count": [
        # If x is a list, then -> calculate len() of the list (or put 0 if it is not a list). Calculate mean for all len() we got
        group_0['selectedIndexes'].apply(lambda x: len(x) if isinstance(x, list) else 0).mean(),
        group_1['selectedIndexes'].apply(lambda x: len(x) if isinstance(x, list) else 0).mean()
    ],
    "Session Finished Count": [
        # Here I count the occurrences of each unique value ("sessionFinished") in event_id, both in group 0 and group 1
        group_0['event_id'].value_counts().get("sessionFinished", 0),
        group_1['event_id'].value_counts().get("sessionFinished", 0)
    ]
}

# Converted the results to a DataFrame, for an easier display
comparison_df = pd.DataFrame(comparison_metrics, index=["Experiment Group 0", "Experiment Group 1"])

pd.set_option('display.max_columns', None)  # Showing all 4 columns
print("Experiment Group Comparison:")
print(comparison_df)
comparison_df.to_csv('experiment_group_comparison.csv', index=True)