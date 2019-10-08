from selenium import webdriver
import os
from util_selenium import *
import csv
import time

import pprint
import urllib.error
import urllib.request

def download_file(url, dst_path):
    try:
        with urllib.request.urlopen(url) as web_file:
            data = web_file.read()
            with open(dst_path, mode='wb') as local_file:
                local_file.write(data)
    except urllib.error.URLError as e:
        print(e)

class KurashiruDetail:
    def __init__(self, driver, page_list, page_num = 707, wait_time = 2):
        self.account = 'koeruqin1998@gmail.com'
        self.password = 'tengteng98'
        self.base_url = 'https://www.kurashiru.com/video_categories/1?page='

        self.page_num = page_num
        self.url_list = []

        self.driver = driver
        self.wait_time = wait_time

        self.mymail = 'koeruqin1998@gmail.com'
        self.mypass = 'tengteng98'

        self.pagelist = page_list

        self.path_dataset = "./data"
        self.saved_num = 0

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

    def get_detail(self):
        for page in self.pagelist:

            uuid = os.path.basename(page)

            datapath = os.path.join(self.path_dataset, uuid)
            
            if os.path.exists(datapath):
                continue
            else:
                if not os.path.exists(datapath):
                    os.mkdir(datapath)
                    print("make dir: ", datapath)

                mydriverget(self.driver, page, self.wait_time)

                #ページが存在しない可能性も
                try:
                    url_video = self.driver.find_element_by_xpath('//*[@id="videos_show"]/div/main/article[1]/article/div[1]/div/video/source[2]').get_attribute('src')
                except:
                    os.rmdir(datapath)
                    continue

                with open(os.path.join(datapath, "video_url.txt"), "w") as f:
                    f.write(url_video)
                
                url_thumb = self.driver.find_element_by_xpath('//*[@id="videos_show"]/div/main/article[1]/article/div[1]/div/video').get_attribute('poster')

                download_file(url_thumb, os.path.join(datapath, "thumbnail.jpg"))

                str_title = self.driver.find_element_by_xpath('//*[@id="videos_show"]/div/main/article[1]/article/div[2]/h1').get_attribute('innerHTML')

                with open(os.path.join(datapath, "title.txt"), "w") as f:
                    f.write(str_title)

                numeric = self.driver.find_elements_by_xpath('//*[@id="videos_show"]/div/main/article[1]/article/div[3]/div[1]/div[*]/span[2]')

                with open(os.path.join(datapath, "numeric.txt"), "w") as f:

                    for data in numeric:
                        f.write(data.get_attribute('innerHTML') + "\n")
                
                ingredients = self.driver.find_elements_by_xpath('//*[@id="videos_show"]/div/main/article[1]/article/section[1]/ul/li[*]/span[1]')
                
                with open(os.path.join(datapath, "ingredients.txt"), "w") as f:
                    for g in ingredients:
                        f.write(g.get_attribute('innerHTML') + "\n")

                instructions = self.driver.find_elements_by_xpath('//*[@id="videos_show"]/div/main/article[1]/article/section[2]/ol/li[*]/span[2]')
                
                with open(os.path.join(datapath, "instructions.txt"), "w") as f:
                    for i in instructions:
                        f.write(i.get_attribute('innerHTML') + "\n")
                
                path_tsukurepo = os.path.join(datapath, "tsukurepo")
                if not os.path.exists(path_tsukurepo):
                    os.mkdir(path_tsukurepo)
                #//*[@id="videos_show"]/div/main/div/article[1]/div/div[1]/div/div[1]/div/img
                tsukurepo = self.driver.find_elements_by_xpath('//*[@id="videos_show"]/div/main/div/article[1]/div/div[1]/div[*]/div[1]/div/img')

                for i in range(len(tsukurepo)):
                    try:
                        download_file(tsukurepo[i].get_attribute('src'), os.path.join(path_tsukurepo, "img" + str(i) + ".jpg"))
                    except:
                        pass

                self.saved_num += 1
                print("processed ", self.saved_num, " ...")

def main():
    with open("./kurasiru_url.txt", "r") as f:
        datalist = f.read().split()

   
    driver = webdriver.Chrome()

    kurasiru = KurashiruDetail(driver, datalist)
    kurasiru.get_detail()


if __name__ == "__main__":
    main()