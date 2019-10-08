from selenium import webdriver
import os
from util_selenium import *
import csv
import time

class Kurashiru:
    def __init__(self, driver, page_num = 707, wait_time = 2):
        self.account = 'koeruqin1998@gmail.com'
        self.password = 'tengteng98'
        self.base_url = 'https://www.kurashiru.com/video_categories/1?page='

        self.page_num = page_num
        self.url_list = []

        self.driver = driver
        self.wait_time = wait_time

        self.mymail = 'koeruqin1998@gmail.com'
        self.mypass = 'tengteng98'

        self.login(self.mymail, self.mypass)

    
    def login(self, mymail, mypass):

        mydriverget(self.driver, self.base_url + "1", self.wait_time)
        login_but = self.driver.find_element_by_xpath('//*[@id="header_app"]/div/div[3]/span/div[2]')
        mydriverclick(login_but, self.driver, self.wait_time)

        field_login = self.driver.find_element_by_xpath('//*[@id="sessions_new"]/div/div[2]/div[1]/div/input')
        field_pass = self.driver.find_element_by_xpath('//*[@id="sessions_new"]/div/div[2]/div[2]/div/input')

        field_login.send_keys(self.mymail)
        field_pass.send_keys(self.mypass)

        login2 = self.driver.find_element_by_xpath('//*[@id="sessions_new"]/div/div[2]/button[1]/div')
        mydriverclick(login2, self.driver, self.wait_time)
    
    def scrape_all_url(self):
        self.url_list = []
        for j in range(self.page_num):
            print(j)

            newurl = self.base_url + str(j)

            mydriverget(self.driver, newurl, self.wait_time)

            ellist = mytrygetel(self.driver.find_elements_by_xpath, '//*[@id="partial_spa"]/div[1]/div/div/main/div[2]/div/li[*]/div/a')

            newlist = []

            for i in range(len(ellist)):
                self.url_list.append(ellist[i].get_attribute('href'))
                newlist.append(ellist[i].get_attribute('href'))
            
            with open("kurasiru_url1.txt", "a") as f:
                for url_ in newlist:
                    f.write(url_ + '\n')

            print("added ", len(ellist), " urls")
            if len(self.url_list) > 0:
                print(self.url_list[-1]) 

        return self.url_list

                                                                    
def main():

    driver = webdriver.Chrome()
    kurasiru = Kurashiru(driver)

    url_list = kurasiru.scrape_all_url()

    print("total url ", len(url_list))

    with open("kurasiru_url2.txt", "w") as f:
        for url in url_list:
            f.write(url + '\n')

if __name__ == '__main__':
    main()