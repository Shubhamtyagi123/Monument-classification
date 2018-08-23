#from imutils import paths
from requests import exceptions 
import argparse
import requests
import cv2 as cv
import os

ap = argparse.ArgumentParser()
ap.add_argument("-q", "--query", required=True, help="search query")
ap.add_argument("-o", "--output", required=True, help="path to directory")
args = vars(ap.parse_args())

# if args.get("query", None) is None:
API_KEY = "8ba3774adf584761a391440c0ac46959"
max_results_search = 200
group_size = 50

search_url = "https://api.cognitive.microsoft.com/bing/v7.0/images/search"

EXCEPTIONS = set([IOError, OSError, exceptions.RequestException, exceptions.HTTPError, exceptions.ConnectionError, exceptions.Timeout])

term = args["query"]
headers = {"Ocp-Apim-Subscription-Key" : API_KEY}
params = {"q": term, "offset": 0, "count": group_size}

print("[Processing] searching Bing API for '{}'".format(term))
search = requests.get(search_url, headers=headers, params=params)
search.raise_for_status()

results = search.json()
estimated_results = min(results["totalEstimatedMatches"], max_results_search)
print("[INFO] {} total results for '{}'".format(estimated_results, term))

total = 0

for offset in range(0, estimated_results, group_size):

	print("[INFO] making requests for group {}-{} of {}...".format(offset, offset+group_size, estimated_results))
	params["offset"] = offset
	search = requests.get(search_url, headers=headers, params=params)
	search.raise_for_status()
	results = search.json()
	print("[INFO] saving images for group {}-{} of {}...".format(offset, offset+group_size, estimated_results))

	for v in results["value"]:

		try:
			print("[INFO] fetching: {}".format(v["contentUrl"]))
			r = requests.get(v["contentUrl"], timeout=30)

			out = v["contentUrl"][v["contentUrl"].rfind("."):]
			_path_ = os.path.sep.join([args["output"], "{}{}".format(str(total).zfill(8), out)])

			file = open(_path_, "wb")
			file.write(r.content)
			file.close()

		except Exception as e:
			if type(e) in EXCEPTIONS:
				print("[INFO] skipping: {}".format(v["contentUrl"]))
				continue

		# out = v["contentUrl"][v["contentUrl"].rfind("."):]
		# _path_ = os.path.sep.join([args["output"], "{}{}".format(str(total).zfill(8), out)])
		
		image = cv.imread(_path_)

		if image is None:
			print("[INFO] deleting: {}".format(_path_))
			os.remove(_path_)
			continue

		total += 1