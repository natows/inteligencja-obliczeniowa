import datetime, math
name = input("Your name: ")
year = int(input("Year of birth: "))
month = int(input("Month of birth: "))
day =  int(input("Day of birth: "))

date = datetime.datetime(year, month, day)
current_date = datetime.datetime.now()
user_days = abs((current_date - date).days)


print(f"Hii {name}, today this is the {user_days} day of your life!")

wspolczynnik_fizyczny = math.sin(2*math.pi/23 * user_days)
wspolczynnik_emocjonalny = math.sin(2*math.pi/28 * user_days)
wspolczynnik_intelektualny = math.sin(2*math.pi/33 * user_days)


def check_next_day(name, days=user_days):
    days += 1
    if name == "physical":
        wspolczynnik_fizyczny = math.sin(2*math.pi/23 * days)
        return wspolczynnik_fizyczny
    elif name == "emotional":
        wspolczynnik_emocjonalny = math.sin(2*math.pi/28 * days)
        return wspolczynnik_emocjonalny
    elif name == "intelectual":
        wspolczynnik_intelektualny = math.sin(2*math.pi/33 * days)
        return wspolczynnik_intelektualny
    else:
        print("Wrong name of biorythm!")
def check_biorythms(name, parametr):
    if (parametr > 0.5):
        print("Congrats your are in a good shape!")
        check = check_next_day(name)
    elif (parametr < -0.5):
        print("Ahh dont be saddd!")
        check = check_next_day(name)
        if check > parametr:
            print(f"Tomorrow your {name} parameter will be better!")
        else:
            print(f"Tomorrow your {name} parameter will be worse :c")

print(f"Your physical coefficient is: {wspolczynnik_fizyczny}")
check_biorythms("physical", wspolczynnik_fizyczny)
print(f"Your emotional coefficient is: {wspolczynnik_emocjonalny}")
check_biorythms("emotional", wspolczynnik_emocjonalny)
print(f"Your intellectual coefficient is: {wspolczynnik_intelektualny}")
check_biorythms("intelectual", wspolczynnik_intelektualny)



#czas spedzony na pisaniu kodu mniej wiecej 20 minut

    
