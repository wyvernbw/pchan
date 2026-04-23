#!/bin/bash

# Color grid from Charmbracelet's Lipgloss
# 8 rows x 14 columns

colors=(
    # Row 0
    "f25d94" "f36c94" "f37994" "f48693" "f49293" "f59e92" "f5a991" "f5b490" "f4bf8f" "f4ca8e" "f3d58c" "f2df8a" "f1ea88" "eff585"
    # Row 1
    "e559a1" "e668a1" "e676a1" "e783a0" "e790a0" "e79b9f" "e7a79f" "e7b29e" "e6bd9d" "e6c89b" "e5d39a" "e4de98" "e3e996" "e1f393"
    # Row 2
    "d855ad" "d865ad" "d873ad" "d881ac" "d98dac" "d999ab" "d8a5ab" "d8b0aa" "d8bca9" "d7c7a7" "d6d2a6" "d5dda4" "d4e8a2" "d2f3a0"
    # Row 3
    "ca50b9" "ca61b8" "ca70b8" "ca7eb7" "c98bb7" "c997b6" "c9a3b5" "c8afb5" "c8bab4" "c7c5b2" "c6d0b1" "c5dcaf" "c3e7ad" "c1f2ab"
    # Row 4
    "bc4cc4" "bb5ec3" "ba6ec2" "ba7cc2" "b989c1" "b895c0" "b8a1bf" "b7adbf" "b6b8be" "b5c4bc" "b4cfbb" "b3dab9" "b1e6b7" "aff1b5"
    # Row 5
    "ac48d0" "aa5bce" "a96bcd" "a879cc" "a787cb" "a693ca" "a59fc9" "a4abc8" "a3b7c7" "a2c2c6" "a0cec4" "9fd9c3" "9de5c1" "9af0bf"
    # Row 6
    "9a43dd" "9758da" "9569d7" "9377d6" "9285d4" "9092d3" "8f9ed2" "8eaad1" "8cb6d0" "8bc1cf" "89cdcd" "87d8cc" "84e4ca" "82efc7"
    # Row 7
    "843fec" "8055e7" "7c67e3" "7a76e0" "7883de" "7690dc" "749cdb" "72a9da" "70b4d8" "6dc0d7" "6bccd6" "68d7d4" "65e3d2" "61eed0"
)

# Function to convert hex to RGB
hex_to_rgb() {
    local hex=$1
    local r=$((16#${hex:0:2}))
    local g=$((16#${hex:2:2}))
    local b=$((16#${hex:4:2}))
    echo "$r;$g;$b"
}

# Function to calculate relative luminance
get_luminance() {
    local hex=$1
    local r=$((16#${hex:0:2}))
    local g=$((16#${hex:2:2}))
    local b=$((16#${hex:4:2}))
    
    # Simple luminance calculation
    local lum=$(( (r * 299 + g * 587 + b * 114) / 1000 ))
    echo $lum
}

# Function to get contrasting text color (black or white)
get_text_color() {
    local hex=$1
    local lum=$(get_luminance "$hex")
    
    # If luminance > 128, use black text; otherwise use white
    if [ $lum -gt 128 ]; then
        echo "0;0;0"
    else
        echo "255;255;255"
    fi
}

echo "Color Grid (14×8) - Charmbracelet Lipgloss"
echo ""

# Print the grid
row=0
col=0
for color in "${colors[@]}"; do
    rgb=$(hex_to_rgb "$color")
    text_color=$(get_text_color "$color")
    
    # Print color with background
    printf "\e[48;2;${rgb}m\e[38;2;${text_color}m %s \e[0m" "$row,$col"
    
    col=$((col + 1))
    
    # New line after 14 columns
    if [ $col -eq 14 ]; then
        echo ""
        col=0
        row=$((row + 1))
    fi
done

echo ""
