"""JLTamp Radio: één zender die dag en nacht doorloopt, voor iedereen dezelfde.

  rotatie.py   welk nummer (pool met gewichten, rotatieregels)
  planning.py  wanneer er gepraat wordt
  mix.py       overvloei, ducking, volume — in numpy, blokje voor blokje
  encoder.py   één ffmpeg die er HLS van maakt
  zender.py    de draden: wachter, voorbereider, speler

De routes staan in routers/radio.py. Aan en uit met de schakelaar `radio`.
"""
