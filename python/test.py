import requests

url = "https://indian-ipo-wallah.p.rapidapi.com/main_ipo_public"

querystring = { "order": "id.desc",  "limit":"5","offset":"0"}

headers = {
    "x-rapidapi-key": "0f4bba4101msh3e1a232cc7d68aap1b10bbjsn06069b63f343",
    "x-rapidapi-host": "indian-ipo-wallah.p.rapidapi.com"
}

response = requests.get(url, headers=headers, params=querystring)

print(response.status_code)
print(response.json())
# listed_