Docker API that summarizes text using BERT. GPU ready.

## Usage
1. `cd docker`
2. `docker-compose up`
3. Wait while the container is being built + the model is downloaded

Now you are ready to call your API from any other machine:

```
import requests

payload = {
    'text': 'Churchill wanted not only to take the fight to the Axis but also to defend the vital trade links to India (through the Suez Canal) and the all-important oil of the Middle East. Some background is needed. All of North Africa was ruled by European powers, mostly by France (and the Vichy regime in particular), with Libya an Italian colony and Egypt technically independent, but in practice ruled by Britain. The British also ruled Palestine (now Israel) directly and Jordan and Iraq indirectly. Syria and Lebanon were under Vichy rule until Commonwealth troops seized them at the same time as suppressing a pro-Axis coup in Iraq in 1941. Finally Ethiopia, since its conquest in 1935–36, was also an Italian possession.',
    'max_length': '200',
	'min_length': '40'
}

response = requests.post('http://localhost:6001/',
    params=payload)

print(response.text)
```

### Parameters
* text
* max_length
* min_length

Note: The service runs on port 6001
