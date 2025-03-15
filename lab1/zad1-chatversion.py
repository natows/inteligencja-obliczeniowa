import datetime
import math

def oblicz_dni_zycia(data_urodzenia):
    dzis = datetime.date.today()
    return (dzis - data_urodzenia).days

def oblicz_biorytm(dni_zycia, cykl):
    return math.sin((2 * math.pi * dni_zycia) / cykl)

def ocen_biorytm(wartosc, nazwa):
    if wartosc > 0.5:
        print(f"Twój {nazwa} biorytm jest wysoki ({wartosc:.2f}). Gratulacje!")
    elif wartosc < -0.5:
        print(f"Twój {nazwa} biorytm jest niski ({wartosc:.2f}). Trzymaj się!")
    else:
        print(f"Twój {nazwa} biorytm jest w normie ({wartosc:.2f}).")

def prognoza_na_jutro(dni_zycia, cykl, wartosc):
    wartosc_jutro = oblicz_biorytm(dni_zycia + 1, cykl)
    if wartosc < -0.5 and wartosc_jutro > wartosc:
        print("Nie martw się. Jutro będzie lepiej!")

def main():
    imie = input("Podaj swoje imię: ")
    rok = int(input("Podaj rok urodzenia: "))
    miesiac = int(input("Podaj miesiąc urodzenia: "))
    dzien = int(input("Podaj dzień urodzenia: "))
    
    data_urodzenia = datetime.date(rok, miesiac, dzien)
    dni_zycia = oblicz_dni_zycia(data_urodzenia)
    
    print(f"Cześć, {imie}! Dziś jest {dni_zycia} dzień Twojego życia.")
    
    fizyczny = oblicz_biorytm(dni_zycia, 23)
    emocjonalny = oblicz_biorytm(dni_zycia, 28)
    intelektualny = oblicz_biorytm(dni_zycia, 33)
    
    ocen_biorytm(fizyczny, "fizyczny")
    prognoza_na_jutro(dni_zycia, 23, fizyczny)
    
    ocen_biorytm(emocjonalny, "emocjonalny")
    prognoza_na_jutro(dni_zycia, 28, emocjonalny)
    
    ocen_biorytm(intelektualny, "intelektualny")
    prognoza_na_jutro(dni_zycia, 33, intelektualny)
    
if __name__ == "__main__":
    main()

#uzyskanie zadania od czata zajelo 1 minute