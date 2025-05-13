#!/bin/bash

# Path to your original fsf file
TEMPLATE_FILE="output.fsf"

# Loop through S01 to S94
for i in {1..94}
do
    # Format the subject number (S01, S02, ..., S94)
    SUBJECT=$(printf "S%02d" $i)

    # Create a modified fsf file for each subject
    FSF_FILE="output_${SUBJECT}.fsf"
    sed "s/S01/${SUBJECT}/g" "$TEMPLATE_FILE" > "$FSF_FILE"
    feat "output_${SUBJECT}.fsf"

    # Run the modified FSF file (if applicable)
    # Uncomment below line if you need to execute something
    # your_command_here "$FSF_FILE"

    echo "Generated and processed: $FSF_FILE"
done
