from selenium import webdriver
from selenium.webdriver.common.keys import Keys

def mydriverget(driver_name, url, wait_time):
    driver_name.get(url)
    driver_name.implicitly_wait(wait_time)

def mydriverclick(click_el, driver_name, wait_time):
    click_el.click()
    driver_name.implicitly_wait(wait_time)

def mydriversearch(search_box, driver_name, search_query, wait_time):
    search_box.send_keys(search_query)
 
    driver_name.implicitly_wait(wait_time)

def mytrygetel(func, *args, **kwargs):
    try:
        result = func(*args, **kwargs)
    except:
        result = None
    
    return result

def myexec(driver_name, wait_time, func, *args, **kwargs):
    try:
        result = func(*args, **kwargs)
        driver_name.implicitly_wait(wait_time)

    except:
        result = None
    
    return result

def scroll_page(driver, wait_time):
    myexec(driver, wait_time, driver.execute_script, "window.scrollTo(0, document.body.scrollHeight);")
    #myexec(driver, wait_time, driver.execute_script, "window.scrollTo(0, window.innerHeight);")

def scroll_page2(driver, wait_time):
    body = driver.find_element_by_tag_name('body')
    for _ in range(10):
        body.send_keys(Keys.PAGE_DOWN)
        driver.implicitly_wait(wait_time)

def mydriverget_newtab(driver_name, url, wait_time):
    return 
