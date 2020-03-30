- - -
EXAMPLE
![output](https://i.ibb.co/DD1mRQW/example.png)
- - -
INPUT

```json
{
  "text": "Le camembert est super :)",
  "top_k": 5
}
```
- - -
EXECUTION
```bash
curl -X POST "https://api-market-place.ai.ovh.net/text-camembert/process" -H "accept: application/json" -H "X-OVH-Api-Key: XXXXXXXX-XXXX-XXXX-XXXX-XXXXXXXXXXXX" -H "Content-Type: application/json" -d '{ "text": "Le camembert est <mask> :)",  "top_k": 5}'
```
- - -

OUTPUT

```json
[
  [
    "Le camembert est délicieux :)",
    0.49091199040412903,
    " délicieux"
  ],
  [
    "Le camembert est excellent :)",
    0.10556956380605698,
    " excellent"
  ],
  [
    "Le camembert est succulent :)",
    0.03453320264816284,
    " succulent"
  ],
  [
    "Le camembert est meilleur :)",
    0.033031098544597626,
    " meilleur"
  ],
  [
    "Le camembert est parfait :)",
    0.03007647767663002,
    " parfait"
  ],
  [
    "Le camembert est bon :)",
    0.02145528607070446,
    " bon"
  ],
  [
    "Le camembert est délicieuse :)",
    0.015332158654928207,
    " délicieuse"
  ],
  [
    "Le camembert est magnifique :)",
    0.012028267607092857,
    " magnifique"
  ],
  [
    "Le camembert est savoureux :)",
    0.009316671639680862,
    " savoureux"
  ],
  [
    "Le camembert est divin :)",
    0.008841886185109615,
    " divin"
  ]
]
```

please refer to swagger documentation for further technical details: [swagger documentation](https://market-place.ai.ovh.net/#!/apis/1e9818cb-d4f7-4028-9818-cbd4f7802840/pages/0061ba6e-2d89-43b1-a1ba-6e2d8943b12b)

* * *
Mascot by [Alix Chagué](https://twitter.com/Alix_Tz) © 2019