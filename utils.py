import pygame


def scale_image(img, factor):
    size = round(img.get_width() * factor), round(img.get_height() * factor)
    return pygame.transform.scale(img, size)


def blit_rotate_center(win, image, top_left, angle):
    rotated_image = pygame.transform.rotate(image, angle)
    new_rect = rotated_image.get_rect(
        center=image.get_rect(topleft=top_left).center)
    win.blit(rotated_image, new_rect.topleft)


def line_intersect(x1, y1, x2, y2, x3, y3, x4, y4):
    # returns a (x, y) tuple or None if there is no intersection
    d = (y4 - y3) * (x2 - x1) - (x4 - x3) * (y2 - y1)
    if d:
        s = ((x4 - x3) * (y1 - y3) - (y4 - y3) * (x1 - x3)) / d
        t = ((x2 - x1) * (y1 - y3) - (y2 - y1) * (x1 - x3)) / d
    else:
        return None
    if not(0 <= s <= 1 and 0 <= t <= 1):
        return None
    x = x1 + s * (x2 - x1)
    y = y1 + s * (y2 - y1)
    return x, y


def line_intersect_front(x1, y1, x2, y2, x3, y3, x4, y4):
    d = (y4 - y3) * (x2 - x1) - (x4 - x3) * (y2 - y1)
    if d:
        s = ((x4 - x3) * (y1 - y3) - (y4 - y3) * (x1 - x3)) / d
        t = ((x2 - x1) * (y1 - y3) - (y2 - y1) * (x1 - x3)) / d
    else:
        return False
    if not(0 <= s and 0 <= t <= 1):
        return False
    x = x1 + s * (x2 - x1)
    y = y1 + s * (y2 - y1)
    return x, y