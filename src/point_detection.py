import cv2
import debug_logging as debug


# TODO: implementáció folytatása a következőképp (az egyes paraméterek finomhangolhatóak):
#   - a kontúrokat szűrni méret alapján: képméret harmadánál kisebbek és századrészénél nagyobbak jók
#   - cirkularitás: perfekt kör esetén 0.08, fogadjuk el a 0.05 felettieket
#   - ha túl kicsi a kép, ezért nem érzékel egy kört se (túl kockás):
#       - input kép nagyítása, medián vagy átlag szűrés, majd pontdetektálás újrakezdése
#   - segédanyag, ha kell: https://www.inf.u-szeged.hu/~tanacs/pyocv/alakjellemzk_szmtsa.html
def count_points(dice):
    dice_mask = cv2.adaptiveThreshold(dice, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY_INV, 15, -15)
    # debugoláshoz, ha bele akarsz nézni, mit csinál a képekkel a progi
    debug.imshowAndPause('dice_mask', dice_mask)
    contours, hierarchy = cv2.findContours(dice_mask, cv2.RETR_LIST, cv2.CHAIN_APPROX_NONE)
    return 0
