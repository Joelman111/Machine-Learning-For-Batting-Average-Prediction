import argparse

parser = argparse.ArgumentParser(description='Predict a players batting average for a given year')
parser.add_argument("Player name", nargs=1, type=string,
                    help="enter a player name (First Last)")
parser.add_argument("Year", nargs=1, type=int, default=2019,
                    help="Enter a year valid for player")

args = parser.parse_args()