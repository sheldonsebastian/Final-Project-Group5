import os

# create list of curl commands
curler_IMFD = ["curl 'https://westeurope1-mediap.svc.ms/transform/zip?cs=fFNQTw' \
  -H 'authority: westeurope1-mediap.svc.ms' \
  -H 'cache-control: max-age=0' \
  -H 'origin: https://esigelec-my.sharepoint.com' \
  -H 'upgrade-insecure-requests: 1' \
  -H 'dnt: 1' \
  -H 'content-type: application/x-www-form-urlencoded' \
  -H 'user-agent: Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/87.0.4280.66 Safari/537.36' \
  -H 'accept: text/html,application/xhtml+xml,application/xml;q=0.9,image/avif,image/webp,image/apng,*/*;q=0.8,application/signed-exchange;v=b3;q=0.9' \
  -H 'sec-fetch-site: cross-site' \
  -H 'sec-fetch-mode: navigate' \
  -H 'sec-fetch-dest: iframe' \
  -H 'accept-language: en-US,en;q=0.9' \
  --data-raw 'zipFileName=48000.zip&guid=2bc4c587-ce75-4730-9d3a-ace92410b9f5&provider=spo&files=%7B%22items%22%3A%5B%7B%22name%22%3A%2248000%22%2C%22size%22%3A0%2C%22docId%22%3A%22https%3A%2F%2Fesigelec-my.sharepoint.com%3A443%2F_api%2Fv2.0%2Fdrives%2Fb%21DgmM8slZ50q8utDSmeka2Yhgv1-Xns1PtKWv-JNK5n3D3VLcUjyJSLDVxCrUSGAG%2Fitems%2F01BH47OV5PQFKMF4WRDNHKRL3BSMYQH3DO%3Fversion%3DPublished%26access_token%3DeyJ0eXAiOiJKV1QiLCJhbGciOiJub25lIn0.eyJhdWQiOiIwMDAwMDAwMy0wMDAwLTBmZjEtY2UwMC0wMDAwMDAwMDAwMDAvZXNpZ2VsZWMtbXkuc2hhcmVwb2ludC5jb21AMzcxY2IxNTYtOTU1OC00Mjg2LWEzY2QtMzA1OTY5OWI4OTBjIiwiaXNzIjoiMDAwMDAwMDMtMDAwMC0wZmYxLWNlMDAtMDAwMDAwMDAwMDAwIiwibmJmIjoiMTYwNjYwODAwMCIsImV4cCI6IjE2MDY2Mjk2MDAiLCJlbmRwb2ludHVybCI6IjJTUUsxTHltWWh6OG1yNmY4YUFhUkJJRnJuT09WVlRnT1FwSlBNcHZJU2M9IiwiZW5kcG9pbnR1cmxMZW5ndGgiOiIxMTgiLCJpc2xvb3BiYWNrIjoiVHJ1ZSIsInZlciI6Imhhc2hlZHByb29mdG9rZW4iLCJzaXRlaWQiOiJaakk0WXpBNU1HVXROVGxqT1MwMFlXVTNMV0pqWW1FdFpEQmtNams1WlRreFlXUTUiLCJuYW1laWQiOiIwIy5mfG1lbWJlcnNoaXB8dXJuJTNhc3BvJTNhYW5vbiMzNDNkYjUwM2JkZmZlNDJjYmI0YzVjZWI3YzFmZGY2ZGUzMWQ2Nzk0MTE4OTI4NGJhYWI2Yjg4NDUyYzA5MjM1IiwibmlpIjoibWljcm9zb2Z0LnNoYXJlcG9pbnQiLCJpc3VzZXIiOiJ0cnVlIiwiY2FjaGVrZXkiOiIwaC5mfG1lbWJlcnNoaXB8dXJuJTNhc3BvJTNhYW5vbiMzNDNkYjUwM2JkZmZlNDJjYmI0YzVjZWI3YzFmZGY2ZGUzMWQ2Nzk0MTE4OTI4NGJhYWI2Yjg4NDUyYzA5MjM1Iiwic2hhcmluZ2lkIjoiUURnWVcvNGxZMGlvZDBHVWw4ODV5USIsInR0IjoiMCIsInVzZVBlcnNpc3RlbnRDb29raWUiOiIyIn0.SnR3bjhYZ2JZcm5qZ1ZjT2dlTDJVK0RSSm8zV1ZPa3dSOFVjYkdrYWpFTT0%22%2C%22isFolder%22%3Atrue%7D%5D%7D&oAuthToken=' \
  --compressed",
               "curl 'https://westeurope1-mediap.svc.ms/transform/zip?cs=fFNQTw' \
  -H 'authority: westeurope1-mediap.svc.ms' \
  -H 'cache-control: max-age=0' \
  -H 'origin: https://esigelec-my.sharepoint.com' \
  -H 'upgrade-insecure-requests: 1' \
  -H 'dnt: 1' \
  -H 'content-type: application/x-www-form-urlencoded' \
  -H 'user-agent: Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/87.0.4280.66 Safari/537.36' \
  -H 'accept: text/html,application/xhtml+xml,application/xml;q=0.9,image/avif,image/webp,image/apng,*/*;q=0.8,application/signed-exchange;v=b3;q=0.9' \
  -H 'sec-fetch-site: cross-site' \
  -H 'sec-fetch-mode: navigate' \
  -H 'sec-fetch-dest: iframe' \
  -H 'accept-language: en-US,en;q=0.9' \
  --data-raw 'zipFileName=60000.zip&guid=b997ef61-2ca0-43a4-b87e-7d6cca69125f&provider=spo&files=%7B%22items%22%3A%5B%7B%22name%22%3A%2260000%22%2C%22size%22%3A0%2C%22docId%22%3A%22https%3A%2F%2Fesigelec-my.sharepoint.com%3A443%2F_api%2Fv2.0%2Fdrives%2Fb%21DgmM8slZ50q8utDSmeka2Yhgv1-Xns1PtKWv-JNK5n3D3VLcUjyJSLDVxCrUSGAG%2Fitems%2F01BH47OVZINSIT2XQ4XBDID72IAU7KPVH3%3Fversion%3DPublished%26access_token%3DeyJ0eXAiOiJKV1QiLCJhbGciOiJub25lIn0.eyJhdWQiOiIwMDAwMDAwMy0wMDAwLTBmZjEtY2UwMC0wMDAwMDAwMDAwMDAvZXNpZ2VsZWMtbXkuc2hhcmVwb2ludC5jb21AMzcxY2IxNTYtOTU1OC00Mjg2LWEzY2QtMzA1OTY5OWI4OTBjIiwiaXNzIjoiMDAwMDAwMDMtMDAwMC0wZmYxLWNlMDAtMDAwMDAwMDAwMDAwIiwibmJmIjoiMTYwNjYwODAwMCIsImV4cCI6IjE2MDY2Mjk2MDAiLCJlbmRwb2ludHVybCI6IjJTUUsxTHltWWh6OG1yNmY4YUFhUkJJRnJuT09WVlRnT1FwSlBNcHZJU2M9IiwiZW5kcG9pbnR1cmxMZW5ndGgiOiIxMTgiLCJpc2xvb3BiYWNrIjoiVHJ1ZSIsInZlciI6Imhhc2hlZHByb29mdG9rZW4iLCJzaXRlaWQiOiJaakk0WXpBNU1HVXROVGxqT1MwMFlXVTNMV0pqWW1FdFpEQmtNams1WlRreFlXUTUiLCJuYW1laWQiOiIwIy5mfG1lbWJlcnNoaXB8dXJuJTNhc3BvJTNhYW5vbiMzNDNkYjUwM2JkZmZlNDJjYmI0YzVjZWI3YzFmZGY2ZGUzMWQ2Nzk0MTE4OTI4NGJhYWI2Yjg4NDUyYzA5MjM1IiwibmlpIjoibWljcm9zb2Z0LnNoYXJlcG9pbnQiLCJpc3VzZXIiOiJ0cnVlIiwiY2FjaGVrZXkiOiIwaC5mfG1lbWJlcnNoaXB8dXJuJTNhc3BvJTNhYW5vbiMzNDNkYjUwM2JkZmZlNDJjYmI0YzVjZWI3YzFmZGY2ZGUzMWQ2Nzk0MTE4OTI4NGJhYWI2Yjg4NDUyYzA5MjM1Iiwic2hhcmluZ2lkIjoiUURnWVcvNGxZMGlvZDBHVWw4ODV5USIsInR0IjoiMCIsInVzZVBlcnNpc3RlbnRDb29raWUiOiIyIn0.SnR3bjhYZ2JZcm5qZ1ZjT2dlTDJVK0RSSm8zV1ZPa3dSOFVjYkdrYWpFTT0%22%2C%22isFolder%22%3Atrue%7D%5D%7D&oAuthToken=' \
  --compressed",
               "curl 'https://westeurope1-mediap.svc.ms/transform/zip?cs=fFNQTw' \
  -H 'authority: westeurope1-mediap.svc.ms' \
  -H 'cache-control: max-age=0' \
  -H 'origin: https://esigelec-my.sharepoint.com' \
  -H 'upgrade-insecure-requests: 1' \
  -H 'dnt: 1' \
  -H 'content-type: application/x-www-form-urlencoded' \
  -H 'user-agent: Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/87.0.4280.66 Safari/537.36' \
  -H 'accept: text/html,application/xhtml+xml,application/xml;q=0.9,image/avif,image/webp,image/apng,*/*;q=0.8,application/signed-exchange;v=b3;q=0.9' \
  -H 'sec-fetch-site: cross-site' \
  -H 'sec-fetch-mode: navigate' \
  -H 'sec-fetch-dest: iframe' \
  -H 'accept-language: en-US,en;q=0.9' \
  --data-raw 'zipFileName=69000.zip&guid=0ec1cd3a-4ca2-4bcd-a865-877ee2527e26&provider=spo&files=%7B%22items%22%3A%5B%7B%22name%22%3A%2269000%22%2C%22size%22%3A0%2C%22docId%22%3A%22https%3A%2F%2Fesigelec-my.sharepoint.com%3A443%2F_api%2Fv2.0%2Fdrives%2Fb%21DgmM8slZ50q8utDSmeka2Yhgv1-Xns1PtKWv-JNK5n3D3VLcUjyJSLDVxCrUSGAG%2Fitems%2F01BH47OV4APW2RS6TA7FBLQQDCC7NITK7B%3Fversion%3DPublished%26access_token%3DeyJ0eXAiOiJKV1QiLCJhbGciOiJub25lIn0.eyJhdWQiOiIwMDAwMDAwMy0wMDAwLTBmZjEtY2UwMC0wMDAwMDAwMDAwMDAvZXNpZ2VsZWMtbXkuc2hhcmVwb2ludC5jb21AMzcxY2IxNTYtOTU1OC00Mjg2LWEzY2QtMzA1OTY5OWI4OTBjIiwiaXNzIjoiMDAwMDAwMDMtMDAwMC0wZmYxLWNlMDAtMDAwMDAwMDAwMDAwIiwibmJmIjoiMTYwNjYwODAwMCIsImV4cCI6IjE2MDY2Mjk2MDAiLCJlbmRwb2ludHVybCI6IjJTUUsxTHltWWh6OG1yNmY4YUFhUkJJRnJuT09WVlRnT1FwSlBNcHZJU2M9IiwiZW5kcG9pbnR1cmxMZW5ndGgiOiIxMTgiLCJpc2xvb3BiYWNrIjoiVHJ1ZSIsInZlciI6Imhhc2hlZHByb29mdG9rZW4iLCJzaXRlaWQiOiJaakk0WXpBNU1HVXROVGxqT1MwMFlXVTNMV0pqWW1FdFpEQmtNams1WlRreFlXUTUiLCJuYW1laWQiOiIwIy5mfG1lbWJlcnNoaXB8dXJuJTNhc3BvJTNhYW5vbiMzNDNkYjUwM2JkZmZlNDJjYmI0YzVjZWI3YzFmZGY2ZGUzMWQ2Nzk0MTE4OTI4NGJhYWI2Yjg4NDUyYzA5MjM1IiwibmlpIjoibWljcm9zb2Z0LnNoYXJlcG9pbnQiLCJpc3VzZXIiOiJ0cnVlIiwiY2FjaGVrZXkiOiIwaC5mfG1lbWJlcnNoaXB8dXJuJTNhc3BvJTNhYW5vbiMzNDNkYjUwM2JkZmZlNDJjYmI0YzVjZWI3YzFmZGY2ZGUzMWQ2Nzk0MTE4OTI4NGJhYWI2Yjg4NDUyYzA5MjM1Iiwic2hhcmluZ2lkIjoiUURnWVcvNGxZMGlvZDBHVWw4ODV5USIsInR0IjoiMCIsInVzZVBlcnNpc3RlbnRDb29raWUiOiIyIn0.SnR3bjhYZ2JZcm5qZ1ZjT2dlTDJVK0RSSm8zV1ZPa3dSOFVjYkdrYWpFTT0%22%2C%22isFolder%22%3Atrue%7D%5D%7D&oAuthToken=' \
  --compressed"]

if __name__ == "__main__":

    for curlURL in curler_IMFD:
        os.system("sudo curl -L -J -O " + curlURL[4:])
