A RESTful BERT server based on tf.Estimator, flask and gunicorn.

To start the server, use 

```bash
bash run.sh
```

After that, get embeddings with

```python
import requests

emb = requests.post("http://0.0.0.0:5000/encode", 
                    json={"texts": ["you always get back to the basics"]}).json()
```
