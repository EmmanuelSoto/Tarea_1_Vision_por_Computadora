from cv2 import cv2, FONT_HERSHEY_COMPLEX_SMALL, resize

files = ["Images\circulo.png", "Images\Circulo2.png", "Images\Cuadrado.png",
         "Images\Cuadrado2.png", "Images\estrella.png", "Images\estrella2.png"]

for file in files:

    image = cv2.imread(file)
    img = resize(image, dsize=(518, 633), interpolation=cv2.INTER_CUBIC)
    img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    _, umbral = cv2.threshold(img_gray, 240, 255, cv2.THRESH_BINARY)
    contornos, _ = cv2.findContours(umbral, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)

    for cont in contornos:
        epsilon = 0.01 * cv2.arcLength(cont, closed=True)
        aprox = cv2.approxPolyDP(cont, epsilon, True)
        cv2.drawContours(img, [aprox], 0, (255, 0, 0), 5)
        x = aprox.ravel()[0]
        y = aprox.ravel()[1]
        if len(aprox) == 4:
            cv2.putText(img, "Cuadrado", (x, y), cv2.FONT_HERSHEY_COMPLEX, 1, (0, 0, 255))
        elif len(aprox) == 10:
            cv2.putText(img, "Estrella", (x, y), cv2.FONT_HERSHEY_COMPLEX, 1, (0, 0, 255))
        else:
            cv2.putText(img, "Circulo", (x, y), cv2.FONT_HERSHEY_COMPLEX, 1, (0, 0, 255))

    cv2.imshow("Figura", img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
