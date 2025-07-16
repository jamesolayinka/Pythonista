import requests
import json
import numpy as np
import logging
from typing import List, Literal

QRNG_API_SOURCES = {
    "anu": "https://qrng.anu.edu.au/API/jsonI.php",
}

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

def fetch_from_anu(length: int = 1, number_type: Literal['uint8', 'uint16'] = 'uint8') -> List[int]:
    """
    Fetch random numbers from the Australian National University QRNG API.
    Args:
        length (int): Number of random numbers to fetch.
        number_type (str): Type of number: 'uint8' or 'uint16'.
    Returns:
        List[int]: List of quantum random numbers.
    """
    
    url = f"{QRNG_API_SOURCES['anu']}?length={length}&type={number_type}"
    try:
        response = requests.get(url, timeout=10)
        response.raise_for_status()
        data = response.json()
        if data.get('success', False):
            return data['data']
        else:
            raise ValueError("QRNG API error: Data fetch unsuccessful.")
    except Exception as e:
        logger.error(f"Error fetching quantum numbers: {e}")
        return []

def save_to_file(data: List[int], file_path: str) -> None:
    """
    Save list of numbers to a json file.
    Args:
        data (List[int]): Random number list.
        file_path (str): Output file path.
    """
    try:
        with open(file_path, 'w') as f:
            json.dump(data, f)
        logger.info(f"Saved {len(data)} numbers to {file_path}")
    except Exception as e:
        logger.error(f"Failed to save data to {file_path}: {e}")

def get_as_numpy_array(data: List[int]) -> np.ndarray:
    """
    Convert list to NumPy array.
    Args:
        data (List[int]): Random number list.
    Returns:
        np.ndarray: NumPy array of quantum numbers.
    """
    return np.array(data)

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Fetch quantum random numbers from ANU QRNG API.")
    parser.add_argument("--length", type=int, default=10, help="Number of random numbers to fetch.")
    parser.add_argument("--type", type=str, default="uint8", choices=["uint8", "uint16"], help="Type of number.")
    parser.add_argument("--save", type=str, help="Optional path to save the numbers as JSON.")
    parser.add_argument("--numpy", action="store_true", help="Print NumPy array output.")

    args = parser.parse_args()

    logger.info(f"Fetching {args.length} quantum random numbers from ANU QRNG...")
    result = fetch_from_anu(length=args.length, number_type=args.type)

    if args.numpy:
        logger.info("NumPy Array Output:")
        print(get_as_numpy_array(result))
    else:
        logger.info("Result:")
        print(result)

    if args.save:
        save_to_file(result, args.save)
