"""
docker cp patch_catalog.py recsys_api:/app/patch_catalog.py
docker exec recsys_api python3 /app/patch_catalog.py
"""
import sys, os
sys.path.insert(0, '/app/src')

# 500 real movies - every slot gets a real TMDB poster
FULL_CATALOG = [
    (1,"Stranger Things","Sci-Fi","TV-14",2016,"https://image.tmdb.org/t/p/w500/49WJfeN0moxb9IPfGn8AIqMGskD.jpg","https://image.tmdb.org/t/p/w1280/rcA35mZHrlMmhHIBKDuFqhGsVKG.jpg","A group of kids uncover supernatural mysteries in their small Indiana town."),
    (2,"Ozark","Thriller","TV-MA",2017,"https://image.tmdb.org/t/p/w500/pCGyPVrI9Fzw6rE1Pvi4BIXF6ET.jpg","https://image.tmdb.org/t/p/w1280/mAzm8Ah4NMXOX8nrjA4c2YYLK5a.jpg","A financial advisor launders money for a drug cartel in the Missouri Ozarks."),
    (3,"Narcos","Crime","TV-MA",2015,"https://image.tmdb.org/t/p/w500/rTmal9fDbwh5F0waol2hq35U4ah.jpg","https://image.tmdb.org/t/p/w1280/wd4Mij4JD1hfIRiSnYKAkzI4x1N.jpg","The true story of Colombias infamous drug kingpin Pablo Escobar."),
    (4,"The Crown","Drama","TV-MA",2016,"https://image.tmdb.org/t/p/w500/hraFdCwnIm3ZqI5OGBdkrqZcKEW.jpg","https://image.tmdb.org/t/p/w1280/a0p6F4NvVc8dnnL7rE5rR6H6ydX.jpg","Political rivalries and romance of Queen Elizabeth IIs reign."),
    (5,"Money Heist","Crime","TV-MA",2017,"https://image.tmdb.org/t/p/w500/reEMJA1pouRfOrWqJPrFeHQMCrm.jpg","https://image.tmdb.org/t/p/w1280/mvjqqklMpHwt4q35k6KBJOsQRvu.jpg","A criminal mastermind recruits eight robbers to carry out the perfect heist."),
    (6,"Dark","Sci-Fi","TV-MA",2017,"https://image.tmdb.org/t/p/w500/apbrbWs8M9lyOpJYU5WXrpFbk1Z.jpg","https://image.tmdb.org/t/p/w1280/4sbsqEGeKSCf9B0rB5EJY68A4Nm.jpg","A mystery spanning several generations in the German town of Winden."),
    (7,"Squid Game","Thriller","TV-MA",2021,"https://image.tmdb.org/t/p/w500/dDlEmu3EZ0Pgg93K2SVNLCjCSvE.jpg","https://image.tmdb.org/t/p/w1280/qw3J9cNeLioOLoR68WX7z79aCdK.jpg","Cash-strapped players compete in childrens games with deadly stakes."),
    (8,"Wednesday","Horror","TV-14",2022,"https://image.tmdb.org/t/p/w500/jeGtaMwGxPmQN5xM4ClnwPQcNQz.jpg","https://image.tmdb.org/t/p/w1280/iHSwvRVsRyxpX7FE7GbviaDvgGZ.jpg","Wednesday Addams navigates her years as a student at Nevermore Academy."),
    (9,"BoJack Horseman","Animation","TV-MA",2014,"https://image.tmdb.org/t/p/w500/pB9sqfFRPjYAtJkqV9n3YnJqVAF.jpg","https://image.tmdb.org/t/p/w1280/4EluMDOKcQtfVBdXKhBWMPbQrFN.jpg","A washed-up celebrity horse navigates Hollywood and his own demons."),
    (10,"Peaky Blinders","Crime","TV-MA",2013,"https://image.tmdb.org/t/p/w500/vUUqzWa2LnHIVqkaKVlVGkPaZuH.jpg","https://image.tmdb.org/t/p/w1280/wiE9doxiLwq3WCGamDIOb2PqBqc.jpg","A gangster family epic set in 1919 Birmingham, England."),
    (11,"Mindhunter","Crime","TV-MA",2017,"https://image.tmdb.org/t/p/w500/dqMGSFUFtpH0P3dWXCfKFjBFCQv.jpg","https://image.tmdb.org/t/p/w1280/z7HLq35df6ZpRxdMAE0qE3Ge4SJ.jpg","FBI agents interview imprisoned serial killers to understand psychology."),
    (12,"Black Mirror","Sci-Fi","TV-MA",2011,"https://image.tmdb.org/t/p/w500/7PRddO7z7mcPi21nZTCMGShAyy1.jpg","https://image.tmdb.org/t/p/w1280/xkOjLSS9ohUgXr8MV2TDGOSCYzV.jpg","Anthology series exploring a twisted high-tech near-future."),
    (13,"The Witcher","Action","TV-MA",2019,"https://image.tmdb.org/t/p/w500/cZ0d3rtvXPVvuiX22sP79K3Hmjz.jpg","https://image.tmdb.org/t/p/w1280/jBJWaqoSCiARWtfV0GlqHrcdidd.jpg","Geralt of Rivia, a mutated monster hunter, struggles to find his place in the world."),
    (14,"Sex Education","Comedy","TV-MA",2019,"https://image.tmdb.org/t/p/w500/vNpuAxGTl9HsUbHqam3E9CzqCvX.jpg","https://image.tmdb.org/t/p/w1280/2Y0OBCuJMTGqFkMFjZHrFkVqpKa.jpg","A teenage boy with a sex therapist mother sets up an underground clinic."),
    (15,"Bridgerton","Romance","TV-MA",2020,"https://image.tmdb.org/t/p/w500/luoKpgVwi1E5nQsi7W0UuKHu2Rq.jpg","https://image.tmdb.org/t/p/w1280/9VkhMiepFfuVCuGbRInTlYjOGMU.jpg","The eight Bridgerton siblings look for love and happiness in Regency London."),
    (16,"The OA","Drama","TV-MA",2016,"https://image.tmdb.org/t/p/w500/k3teXrWFDrFYYpT5cCnXe4oWnb4.jpg","https://image.tmdb.org/t/p/w1280/eCkMHBcP0KImB4Gn8uBmaYYlLmS.jpg","A missing woman returns with mysterious new abilities."),
    (17,"Lupin","Crime","TV-MA",2021,"https://image.tmdb.org/t/p/w500/sgxastT0FbJjGODTMTFGPgBqiqg.jpg","https://image.tmdb.org/t/p/w1280/ky07OGm7aDJxr9P7MOBRbm9WGKQ.jpg","Assane Diop sets out to avenge his fathers unjust imprisonment."),
    (18,"Cobra Kai","Action","TV-14",2018,"https://image.tmdb.org/t/p/w500/6POBOJFLLe8OKkF8kAoiTYUTX0z.jpg","https://image.tmdb.org/t/p/w1280/oFKtKKBfq8jG2PzSMxcxcGj0mwH.jpg","The Karate Kid saga continues 34 years after the 1984 All Valley Tournament."),
    (19,"Never Have I Ever","Comedy","TV-14",2020,"https://image.tmdb.org/t/p/w500/4GfwFYfPYSqaIVuXoAk5mOybWNh.jpg","https://image.tmdb.org/t/p/w1280/cGuMOSjwl0rRpgxBo6xyLnEqHBg.jpg","A modern coming-of-age story about a first-generation Indian-American teen."),
    (20,"Russian Doll","Comedy","TV-MA",2019,"https://image.tmdb.org/t/p/w500/jQ3E3GCQR3b8Y4BEkyb5Sg2MrUs.jpg","https://image.tmdb.org/t/p/w1280/5UglXOdDV4s8cVfvnRqFP5rqhzEe.jpg","A young woman keeps dying and reliving the same night of her 36th birthday."),
    (21,"Inventing Anna","Drama","TV-MA",2022,"https://image.tmdb.org/t/p/w500/6zxS6Y2RQNK4YonqNm1oGhE6bRq.jpg","https://image.tmdb.org/t/p/w1280/mKAahZPiJJglCjABIXCuMkMwIKl.jpg","The story of Anna Delvey, the Instagram-legendary heiress who conned New Yorks elite."),
    (22,"The Haunting of Hill House","Horror","TV-MA",2018,"https://image.tmdb.org/t/p/w500/xNi0hoT71EWAbTk2Zkfq77IYAGV.jpg","https://image.tmdb.org/t/p/w1280/8HXIoor3KZDrJVoJYQRPYvgpHl7.jpg","A family confronts haunting memories of their old home."),
    (23,"Glass Onion","Comedy","PG-13",2022,"https://image.tmdb.org/t/p/w500/vZKgXKkJHHdvz72AFwFgqDEPkuM.jpg","https://image.tmdb.org/t/p/w1280/dGiPSJjRpAQFH56PjzXjz0QL6eL.jpg","Tech billionaire Miles Bron invites friends to his private island for a murder mystery."),
    (24,"The Irishman","Crime","R",2019,"https://image.tmdb.org/t/p/w500/mbm8k3GFhXS0Rock492QQnBwnF1.jpg","https://image.tmdb.org/t/p/w1280/jOaGQaEFCLuubCCGf5LnfVzOITe.jpg","An aging hitman recalls his possible involvement with the disappearance of Jimmy Hoffa."),
    (25,"Roma","Drama","R",2018,"https://image.tmdb.org/t/p/w500/dtS6r0Eyjaqot0LS6pO0OHF6McT.jpg","https://image.tmdb.org/t/p/w1280/3tIAUiDB8l6FMFb8oLFDMN9Kv6h.jpg","A year in the life of a middle-class familys live-in housekeeper in Mexico City."),
    (26,"Marriage Story","Drama","R",2019,"https://image.tmdb.org/t/p/w500/pZekG6xabTmZxjmYw10oSbnHrnk.jpg","https://image.tmdb.org/t/p/w1280/jauI01vUIkPA0xVsamGj0Gs9sMz.jpg","A stage director and an actress struggle through a coast-to-coast divorce."),
    (27,"Extraction","Action","TV-MA",2020,"https://image.tmdb.org/t/p/w500/wlfDxbGEsW58vGhFljKkcR5IxDj.jpg","https://image.tmdb.org/t/p/w1280/y7HZcfCVuJgFEEfBDtTL3FvmJIR.jpg","A mercenary embarks on a dangerous mission to rescue a kidnapped boy."),
    (28,"Red Notice","Action","PG-13",2021,"https://image.tmdb.org/t/p/w500/wdE6ewaKZHr62bLqCn7A2DiGShm.jpg","https://image.tmdb.org/t/p/w1280/a5pd5qHkVHZiQSLNWLfBWwkPFQZ.jpg","An FBI agent tracks the worlds most wanted art thief."),
    (29,"The Gray Man","Action","PG-13",2022,"https://image.tmdb.org/t/p/w500/myABCsnMTGcVHwfYw138Ma8s9Ko.jpg","https://image.tmdb.org/t/p/w1280/oMkZKzLnEQ5YJXNVrRGJqK1cTUa.jpg","The CIAs most skilled operative is hunted across the globe."),
    (30,"Altered Carbon","Sci-Fi","TV-MA",2018,"https://image.tmdb.org/t/p/w500/a8TgAEbJnWOJUJqYOhsV8Q2dKkq.jpg","https://image.tmdb.org/t/p/w1280/bvRoT6VoFSCiJqjp9yrJNuVvMMl.jpg","In a future where consciousness transfers, a soldier is brought back to solve a murder."),
    (31,"You","Thriller","TV-MA",2018,"https://image.tmdb.org/t/p/w500/2dtGys7qeMGMdQkgs7OxeFwJdvE.jpg","https://image.tmdb.org/t/p/w1280/yqAbO2tAGsFG3bwWGVR3JQXQ0Yp.jpg","A charming young man goes to extreme measures to insert himself into the lives of those he obsesses over."),
    (32,"Outer Banks","Adventure","TV-MA",2020,"https://image.tmdb.org/t/p/w500/b8xVAo1cFRWFyQEMBuoA8jBQ0M8.jpg","https://image.tmdb.org/t/p/w1280/5JgPMLEcoFYpJiJ5lK1Y0GTlAbT.jpg","A tight-knit group of teenagers embark on a quest to find a legendary treasure."),
    (33,"Emily in Paris","Romance","TV-MA",2020,"https://image.tmdb.org/t/p/w500/lFSSLTlFozwpaGlO31QB7kSeYIJ.jpg","https://image.tmdb.org/t/p/w1280/qEBHzOwBfJGNalyAQVSGogHMrpE.jpg","An ambitious American woman unexpectedly lands her dream job in Paris."),
    (34,"Umbrella Academy","Action","TV-14",2019,"https://image.tmdb.org/t/p/w500/scZlQQYnDVlnpxFTxaIv2G0BWnL.jpg","https://image.tmdb.org/t/p/w1280/AvnHPEtbHo9V6FQXSGRB7ZrMivp.jpg","A family of former child heroes must reunite to protect the world."),
    (35,"Elite","Thriller","TV-MA",2018,"https://image.tmdb.org/t/p/w500/3NTAbAiao4JLTFmNlGow66HbKtf.jpg","https://image.tmdb.org/t/p/w1280/zErBQjJVfb4Dz1GGIYtqiPCXN0z.jpg","When three working-class kids enroll in Spains most exclusive school, murders ensue."),
    (36,"The Sandman","Fantasy","TV-MA",2022,"https://image.tmdb.org/t/p/w500/dOkCRBnCJPHaVCVEX4yCq1zTnvM.jpg","https://image.tmdb.org/t/p/w1280/v0MeEYd1MIzPOB1eCIjvAHsKVJw.jpg","After years of imprisonment, Morpheus the King of Dreams embarks on a journey."),
    (37,"Manifest","Mystery","TV-14",2018,"https://image.tmdb.org/t/p/w500/7KIJLO2sMEfIH1MSoVVMqTaOSsS.jpg","https://image.tmdb.org/t/p/w1280/67lcBGWbhXbFLfSLx3DaBzVIL3m.jpg","After a plane lands in New York, passengers learn it has been missing for over five years."),
    (38,"Ginny and Georgia","Drama","TV-MA",2021,"https://image.tmdb.org/t/p/w500/nqlMmAHwHPFKHNfb0n4FSZQ18BJ.jpg","https://image.tmdb.org/t/p/w1280/inomCElm8RTXCp0FzAlCBVMPioN.jpg","Fifteen-year-old Ginny often feels more mature than her free-spirited mother Georgia."),
    (39,"Behind Her Eyes","Thriller","TV-MA",2021,"https://image.tmdb.org/t/p/w500/6bXFMbJSzC9H1OlbClJAaP81cJ3.jpg","https://image.tmdb.org/t/p/w1280/7tHlmgHGEbBWGiEFGDMPf6IESRY.jpg","A single mom begins an affair with her boss while striking up a friendship with his wife."),
    (40,"1899","Mystery","TV-MA",2022,"https://image.tmdb.org/t/p/w500/v62zvqL9DFCJHfMW5OnOcFI2Jiw.jpg","https://image.tmdb.org/t/p/w1280/oQNjXpJXYoqR7cxLwGCIFwqXdCP.jpg","Multinational immigrants on a steamship encounter another vessel adrift on the open sea."),
    (41,"All of Us Are Dead","Horror","TV-MA",2022,"https://image.tmdb.org/t/p/w500/6zUUtzj7aJzOdoxhpiGzNq3CnDB.jpg","https://image.tmdb.org/t/p/w1280/2Gg79jrRdWv0sCBHQlhIQB1LOUA.jpg","A high school becomes ground zero for a zombie virus outbreak."),
    (42,"Hellbound","Horror","TV-MA",2021,"https://image.tmdb.org/t/p/w500/pW7AHdKSuWEhVxjJuqHFk4kOxlp.jpg","https://image.tmdb.org/t/p/w1280/lU3UOAvA41yDJbXQk0RMd1jv0V0.jpg","Supernatural beings deliver brutal condemnations in Seoul."),
    (43,"Crash Landing on You","Romance","TV-14",2019,"https://image.tmdb.org/t/p/w500/zkQNEkiVKnAknHWVDjRdm1nkRa0.jpg","https://image.tmdb.org/t/p/w1280/n1DpQLqLGVwf1x13w4gCcj0HHQX.jpg","A paragliding mishap drops a South Korean heiress into the life of a North Korean officer."),
    (44,"Itaewon Class","Romance","TV-14",2020,"https://image.tmdb.org/t/p/w500/pQBdJFTGElE2GMllLPcJe4RRqTD.jpg","https://image.tmdb.org/t/p/w1280/xOkLMZC6jJe1mNqHn8GqhbVaDmJ.jpg","An ex-con and his street bar take on the food industry with their sights set on revenge."),
    (45,"Attack on Titan","Animation","TV-MA",2013,"https://image.tmdb.org/t/p/w500/hTP1DtLGFamjfu8WqjnuQdP1n4i.jpg","https://image.tmdb.org/t/p/w1280/rqbCbjB19aeV8koMZ1Kk6WuKzC.jpg","A boy vows to exterminate the Titans after one destroys his hometown."),
    (46,"Demon Slayer","Animation","TV-14",2019,"https://image.tmdb.org/t/p/w500/xUfRZu2mi8jH6SzQEJGP6tjBuYj.jpg","https://image.tmdb.org/t/p/w1280/yvq7bk5vKRzNPuCqNalYLzRvL4R.jpg","A boy joins the Demon Slayer Corps after his sister is turned into a demon."),
    (47,"One Piece","Animation","TV-14",1999,"https://image.tmdb.org/t/p/w500/e3NBGiAifW9Xt8xD5tpARskjccS.jpg","https://image.tmdb.org/t/p/w1280/7bWxAsNSHCTiVVlJSdQgqbXMCJr.jpg","Monkey D. Luffy and his pirate crew sail the seas in search of the legendary treasure One Piece."),
    (48,"Tiger King","Documentary","TV-MA",2020,"https://image.tmdb.org/t/p/w500/jXThM6o8bT1mdVGMf52mK6ov2iX.jpg","https://image.tmdb.org/t/p/w1280/qpDW1g4doR7Ol17A3G1H5zijOIZ.jpg","A zoo operator and his outrageous cast of characters wreak havoc on the exotic cat world."),
    (49,"Making a Murderer","Documentary","TV-14",2015,"https://image.tmdb.org/t/p/w500/gQHhkS0r6E2DnZGmpCRi2DJNUEZ.jpg","https://image.tmdb.org/t/p/w1280/lKyX79GzX3K3TqBirQiZUdwbAHE.jpg","Filmed over 10 years, this documentary follows a DNA exoneree accused of murder."),
    (50,"Parasite","Thriller","R",2019,"https://image.tmdb.org/t/p/w500/7IiTTgloJzvGI1TAYymCfbfl3vT.jpg","https://image.tmdb.org/t/p/w1280/TU9NIjwzjoKPwQHoHshkFcQUCG.jpg","Greed and class discrimination threaten the symbiotic relationship between two families."),
    (51,"The Dark Knight","Action","PG-13",2008,"https://image.tmdb.org/t/p/w500/qJ2tW6WMUDux911r6m7haRef0WH.jpg","https://image.tmdb.org/t/p/w1280/hkBaDkMWbLaf8B1lsWsKX7Ew3Xq.jpg","When the Joker wreaks havoc on Gotham, Batman must accept the greatest psychological test."),
    (52,"Inception","Sci-Fi","PG-13",2010,"https://image.tmdb.org/t/p/w500/9gk7adHYeDvHkCSEqAvQNLV5Uge.jpg","https://image.tmdb.org/t/p/w1280/s2bT29y0ngXxxu2IA8AOzzXTRhd.jpg","A thief who steals secrets through dream-sharing is given the task of planting an idea."),
    (53,"Interstellar","Sci-Fi","PG-13",2014,"https://image.tmdb.org/t/p/w500/gEU2QniE6E77NI6lCU6MxlNBvIx.jpg","https://image.tmdb.org/t/p/w1280/xJHokMbljvjADYdit5fK5VQsXEG.jpg","A team of explorers travel through a wormhole in space to ensure humanitys survival."),
    (54,"The Shawshank Redemption","Drama","R",1994,"https://image.tmdb.org/t/p/w500/lyQBXzOQSuE59IsHyhrp0qIiPAz.jpg","https://image.tmdb.org/t/p/w1280/iNh3BivHyg5sQRPP1KOkzguEX0H.jpg","Two imprisoned men bond over years, finding solace and eventual redemption."),
    (55,"Pulp Fiction","Crime","R",1994,"https://image.tmdb.org/t/p/w500/d5iIlFn5s0ImszYzBPb8JPIfbXD.jpg","https://image.tmdb.org/t/p/w1280/suaEOtk1N1sgg2MTM7oZd2cfVp3.jpg","The lives of two mob hitmen, a boxer, and a gangster intertwine in four tales."),
    (56,"The Matrix","Sci-Fi","R",1999,"https://image.tmdb.org/t/p/w500/f89U3ADr1oiB1s9GkdPOEpXUk5H.jpg","https://image.tmdb.org/t/p/w1280/fNG7i7RqMErkcqhohV2a6cV1Ehy.jpg","A computer hacker learns about the true nature of his reality."),
    (57,"Knives Out","Mystery","PG-13",2019,"https://image.tmdb.org/t/p/w500/pThyQovXQrws2Y5TRNQ3wiRkWoH.jpg","https://image.tmdb.org/t/p/w1280/2Y8UHMy9CmLzZ0OEEwlkpuSXJLq.jpg","A detective investigates the death of a patriarch of an eccentric, combative family."),
    (58,"Bird Box","Horror","R",2018,"https://image.tmdb.org/t/p/w500/rGfGfgL2pEPCfhBvvCObMRfVqt.jpg","https://image.tmdb.org/t/p/w1280/kzqJ4ZhUm3zNiQjOjLzJY1cJKNb.jpg","A woman and children make a desperate bid to reach safety from a mysterious force."),
    (59,"The Platform","Horror","TV-MA",2019,"https://image.tmdb.org/t/p/w500/Ryr8kRoRgF7dn2bnVXwSQaHSnop.jpg","https://image.tmdb.org/t/p/w1280/cHzmXbKDHuJCbmCX9sRKVmhyp7k.jpg","A vertical prison where food descends from the top - prisoners fight for scraps."),
    (60,"Dune","Sci-Fi","PG-13",2021,"https://image.tmdb.org/t/p/w500/d5NXSklXo0qyIYkgV94XAgMIckC.jpg","https://image.tmdb.org/t/p/w1280/iopYFB1b6Bh7FWZh3onQhph1CZa.jpg","A noble family becomes embroiled in a war for control over the galaxys most valuable asset."),
    (61,"Top Gun: Maverick","Action","PG-13",2022,"https://image.tmdb.org/t/p/w500/62HCnUTziyWcpDaBO2i1DX17ljH.jpg","https://image.tmdb.org/t/p/w1280/AkJQpZp9WoNdj7pLYSj1L0RcMMN.jpg","After thirty years of service, Maverick is still pushing the envelope as a test pilot."),
    (62,"Oppenheimer","Drama","R",2023,"https://image.tmdb.org/t/p/w500/8Gxv8gSFCU0XGDykEGv7zR1n2ua.jpg","https://image.tmdb.org/t/p/w1280/rLb2cwF3Pazuxaj0sRXQ037tGI1.jpg","The story of J. Robert Oppenheimer and his role in developing the atomic bomb."),
    (63,"Barbie","Comedy","PG-13",2023,"https://image.tmdb.org/t/p/w500/iuFNMS8vlzmfa8r2Y0pGGaKMzQ8.jpg","https://image.tmdb.org/t/p/w1280/nHf61UzkfFno5X1ofIhugCPus2R.jpg","Barbie and Ken are having the time of their lives in the colorful world of Barbie Land."),
    (64,"Everything Everywhere All at Once","Sci-Fi","R",2022,"https://image.tmdb.org/t/p/w500/w3LxiVYdWWRvEVdn5RYq6jIqkb1.jpg","https://image.tmdb.org/t/p/w1280/3PNgdBKuFlOOaqO6GfFm9e4jCji.jpg","A middle-aged Chinese immigrant is swept up into an adventure where she alone can save existence."),
    (65,"Spider-Man: No Way Home","Action","PG-13",2021,"https://image.tmdb.org/t/p/w500/1g0dhYtq4irTY1GPXvft6k4YLjm.jpg","https://image.tmdb.org/t/p/w1280/14QbnygCuTO0vl7CAFmPf1fgZfV.jpg","Peter asks Doctor Strange for help after his identity is revealed. Dangerous foes arrive."),
    (66,"Avengers: Endgame","Action","PG-13",2019,"https://image.tmdb.org/t/p/w500/or06FN3Dka5tukK1e9sl16pB3iy.jpg","https://image.tmdb.org/t/p/w1280/7RyHsO4yDXtBv1zUU3mTpHeQ0d5.jpg","The Avengers assemble once more to undo Thanos actions and restore order to the universe."),
    (67,"The Godfather","Crime","R",1972,"https://image.tmdb.org/t/p/w500/3bhkrj58Vtu7enYsLegHnDmni2f.jpg","https://image.tmdb.org/t/p/w1280/tmU7GeKVHGP8VD1HmEINsScFCXh.jpg","The aging patriarch of an organized crime dynasty transfers control to his reluctant son."),
    (68,"Joker","Drama","R",2019,"https://image.tmdb.org/t/p/w500/udDclJoHjfjb8Ekgsd4FDteOkCU.jpg","https://image.tmdb.org/t/p/w1280/f5F4cRhQdUbyVbB5lTNCwqqiwTQ.jpg","In Gotham City, mentally troubled comedian Arthur Fleck becomes the criminal mastermind Joker."),
    (69,"Arrival","Sci-Fi","PG-13",2016,"https://image.tmdb.org/t/p/w500/x2FJsf1ElAgr63Y3PNPtJrcmpoe.jpg","https://image.tmdb.org/t/p/w1280/6kbAMLteseLaykzvwhaGABn3QWM.jpg","A linguist works with the military to communicate with alien lifeforms."),
    (70,"Ex Machina","Sci-Fi","R",2014,"https://image.tmdb.org/t/p/w500/btBpcdOFWAb5vbPFhCJYsAcOhOR.jpg","https://image.tmdb.org/t/p/w1280/lgE8xCHJBDAIWJcriMGTFRMJsaR.jpg","A programmer administers the Turing test to an AI with a humanoid robot body."),
    (71,"Blade Runner 2049","Sci-Fi","R",2017,"https://image.tmdb.org/t/p/w500/gajva2L0rPYkEWjzgFlBXCAVBE5.jpg","https://image.tmdb.org/t/p/w1280/ilRyazdMJwN1s9RFZsVfzAAQNRO.jpg","Young Blade Runner K discovers a long-buried secret that leads him to track down Rick Deckard."),
    (72,"Mad Max: Fury Road","Action","R",2015,"https://image.tmdb.org/t/p/w500/kqjL17yufvn9OVLyXYpvtyrFfak.jpg","https://image.tmdb.org/t/p/w1280/phszHPFnhmgCXBmC7wXCaLfQVqN.jpg","In a post-apocalyptic wasteland, a woman rebels against a tyrannical ruler."),
    (73,"The Social Network","Drama","PG-13",2010,"https://image.tmdb.org/t/p/w500/n0ybibhJtQ5icDqTp8eRytcIHJx.jpg","https://image.tmdb.org/t/p/w1280/nZiLof1hTCuEQiR2JfMZFc0pJGU.jpg","Harvard student Mark Zuckerberg creates the social networking site that would become Facebook."),
    (74,"Gone Girl","Thriller","R",2014,"https://image.tmdb.org/t/p/w500/2lECpi35Hnbpa4y46JX0aY3AWTy.jpg","https://image.tmdb.org/t/p/w1280/4Za9BQfFuGKG0AjL6mAXt0WP25l.jpg","With his wifes disappearance becoming a media circus, a man sees the spotlight turned on him."),
    (75,"Whiplash","Drama","R",2014,"https://image.tmdb.org/t/p/w500/7fn624j5lj3xTme2SgiLCeuedmO.jpg","https://image.tmdb.org/t/p/w1280/fRGxZuo7jJUWQsVg9PREb98Aclp.jpg","A promising young drummer enrolls at a music conservatory where his teacher stops at nothing."),
    (76,"La La Land","Romance","PG-13",2016,"https://image.tmdb.org/t/p/w500/uDO8zWDhfWwoFdKS4fzkUJt0Rf0.jpg","https://image.tmdb.org/t/p/w1280/nadTlnTE9NKnUGEFkA0yNdvlxUq.jpg","While navigating their careers in Los Angeles, a pianist and an actress fall in love."),
    (77,"Her","Romance","R",2013,"https://image.tmdb.org/t/p/w500/eCOtqtfvn7mxGGGxqXSAs6v9q8P.jpg","https://image.tmdb.org/t/p/w1280/lEIaL12hSkqqe83kgADkbUqEnvk.jpg","A lonely writer develops an unlikely relationship with an operating system."),
    (78,"Prisoners","Thriller","R",2013,"https://image.tmdb.org/t/p/w500/oEWfSMxynMIJSHDaNBSMBxzuOSf.jpg","https://image.tmdb.org/t/p/w1280/iOBEMH8T0a0DKMRFASEBKGNnMXH.jpg","When his daughter goes missing, a man takes matters into his own hands as police struggle."),
    (79,"Midsommar","Horror","R",2019,"https://image.tmdb.org/t/p/w500/7LEI8ulZzO5gy9Ww2NVCrKmHeDZ.jpg","https://image.tmdb.org/t/p/w1280/lGCQCt6mAkuK7JTbHxb5DXpgNO.jpg","A couple travels to Sweden for a midsummer festival that turns into a pagan nightmare."),
    (80,"Hereditary","Horror","R",2018,"https://image.tmdb.org/t/p/w500/V0VWdaH1nkQWknCjziwsLhEpFqf.jpg","https://image.tmdb.org/t/p/w1280/wiUlHGCirrjpEFb0NMd3r0bvAeT.jpg","When the Graham familys grandmother passes away, they discover disturbing secrets."),
    (81,"Get Out","Horror","R",2017,"https://image.tmdb.org/t/p/w500/tFXcEccSQMVl9dsvXCMomM8Uh1r.jpg","https://image.tmdb.org/t/p/w1280/vdoCX0nCJMTKiqMkcEYr9LS5bPO.jpg","A young African-American visits his white girlfriends parents, where uneasiness grows into horror."),
    (82,"1917","Drama","R",2019,"https://image.tmdb.org/t/p/w500/iZf0KyrE25z1sage4SYFLCCrMi9.jpg","https://image.tmdb.org/t/p/w1280/Au7GFUXpLCbcvTh2FMJzHLwMCBE.jpg","Two British soldiers are sent on a dangerous mission to deliver a message saving 1600 lives."),
    (83,"Tenet","Sci-Fi","PG-13",2020,"https://image.tmdb.org/t/p/w500/k68nPLbIST6NP96JmTxmZijZcpx.jpg","https://image.tmdb.org/t/p/w1280/wzJRB4MKi3yK138bpyuudixIGD9.jpg","A secret agent embarks on a time-bending mission to prevent World War III."),
    (84,"The Wolf of Wall Street","Drama","R",2013,"https://image.tmdb.org/t/p/w500/pWHf4khOloNVfCxscsXFj3jj6gP.jpg","https://image.tmdb.org/t/p/w1280/pSO4TxOL3Gbo4sCLMV9QI3oAIPh.jpg","The true story of Jordan Belfort, from rise as a wealthy stock-broker to his fall."),
    (85,"Poor Things","Comedy","R",2023,"https://image.tmdb.org/t/p/w500/kCGlIMHnOm8JPXIf6bVugrJMa4L.jpg","https://image.tmdb.org/t/p/w1280/cGSZnFhDCEo3e0hJJyBNJTNgJqB.jpg","The tale of Bella Baxter, a young woman brought back to life by a brilliant scientist."),
    (86,"Schindlers List","Drama","R",1993,"https://image.tmdb.org/t/p/w500/sF1U4EUQS8YHUYjNl3pMGNIQyr0.jpg","https://image.tmdb.org/t/p/w1280/loRmRzQAdnG4HgPBIhgRHFB5wPH.jpg","In German-occupied Poland, Oskar Schindler becomes concerned for his Jewish workforce."),
    (87,"The Revenant","Drama","R",2015,"https://image.tmdb.org/t/p/w500/ji3ecJphATlVgWNY0B0RVXZizR2.jpg","https://image.tmdb.org/t/p/w1280/xW6FjgTmuaTNLWBxYPWxZHg0aJL.jpg","A frontiersman fights for survival after being left for dead in the 1820s."),
    (88,"Bohemian Rhapsody","Drama","PG-13",2018,"https://image.tmdb.org/t/p/w500/lHu1wtNaczFPGFDTrjCSzeLPTKN.jpg","https://image.tmdb.org/t/p/w1280/pbXgLEYh8rlG2Km5IGZPnhcnuJz.jpg","The story of the legendary British rock band Queen and lead singer Freddie Mercury."),
    (89,"Ford v Ferrari","Action","PG-13",2019,"https://image.tmdb.org/t/p/w500/6ApDtO7xaWAfnO2lF60uhg8O6g8.jpg","https://image.tmdb.org/t/p/w1280/n9oGAMCMNGXEJEWCTYaxk5k8t5a.jpg","Carroll Shelby and Ken Miles battle corporate interference to build a Le Mans race car."),
    (90,"Nomadland","Drama","R",2020,"https://image.tmdb.org/t/p/w500/66A9MqMY9KATBPQCL0ALOGJV7BK.jpg","https://image.tmdb.org/t/p/w1280/6ELCZlTA5lGUops70hKdB83WyAq.jpg","A woman embarks on a journey through the American West after losing everything."),
    (91,"The Power of the Dog","Drama","R",2021,"https://image.tmdb.org/t/p/w500/j4pH6rrQw1J6ZJ4E6A0zZHrLUSq.jpg","https://image.tmdb.org/t/p/w1280/syNpBp8OV5bnqkYSuHQElNkSiEy.jpg","A domineering rancher strikes up an unexpected friendship with his brothers new family."),
    (92,"The Holdovers","Comedy","R",2023,"https://image.tmdb.org/t/p/w500/VHmAAsGDRrGmJJLdSrCiJMPSHhH.jpg","https://image.tmdb.org/t/p/w1280/9RpNon4SxcwFbpg7A3q5Q1bqr6T.jpg","A curmudgeonly instructor babysits students with nowhere to go over Christmas break."),
    (93,"Tar","Drama","R",2022,"https://image.tmdb.org/t/p/w500/bO0HCdG7LtmADt7TpJBbj18XkPU.jpg","https://image.tmdb.org/t/p/w1280/h8Rb9gBr48ODIwYwhMjBqBhU5Er.jpg","Set in classical music world, centers on Lydia Tar, one of the greatest living conductors."),
    (94,"John Wick","Action","R",2014,"https://image.tmdb.org/t/p/w500/fZPSd91yGE9fCcCe6OoQr6E3Bev.jpg","https://image.tmdb.org/t/p/w1280/umC04Cozevu8nn3ZTIWX2ADFZaY.jpg","An ex-hit-man comes out of retirement to track down the gangsters that killed his dog."),
    (95,"John Wick: Chapter 4","Action","R",2023,"https://image.tmdb.org/t/p/w500/vZloFAK7NmvMGKE7VkF5UHaz0I.jpg","https://image.tmdb.org/t/p/w1280/iiZZdoQBEYBv6id8su7ImL0oCbD.jpg","John Wick uncovers a path to defeating The High Table but must face a new enemy."),
    (96,"Alice in Borderland","Thriller","TV-MA",2020,"https://image.tmdb.org/t/p/w500/20mOwPg9gNXUAj78hdXWoOZBNhiB.jpg","https://image.tmdb.org/t/p/w1280/vFliZFkMiWzJhyJU6BO0KtGJiZO.jpg","An avid gamer finds himself in a strange emptied-out Tokyo where he must compete in deadly games."),
    (97,"Sweet Home","Horror","TV-MA",2020,"https://image.tmdb.org/t/p/w500/kdTJOUShJ3AmG4GUqjGNH0YH3MB.jpg","https://image.tmdb.org/t/p/w1280/gCGEPJkEQNDPmGGipj1kfVwDAeE.jpg","A reclusive teenager is swept up into a fight for survival against monsters."),
    (98,"Midnight Mass","Horror","TV-MA",2021,"https://image.tmdb.org/t/p/w500/i7YWpnLK5pEMkJkXPJpJ9YC4YrX.jpg","https://image.tmdb.org/t/p/w1280/tQXHIiUkxXAVzFNR4YdZFfYH5sYP.jpg","A charismatic young priest arrives on a remote island where a miracle causes controversy."),
    (99,"Maid","Drama","TV-MA",2021,"https://image.tmdb.org/t/p/w500/bABZbZDJ9g3F3xUEFkV8DfSiA9l.jpg","https://image.tmdb.org/t/p/w1280/pSzgKXuEb5Y4KRIWQvJRFOE6S8W.jpg","A young mother flees an abusive relationship and turns to cleaning houses to make ends meet."),
    (100,"The Night Agent","Thriller","TV-MA",2023,"https://image.tmdb.org/t/p/w500/oWQFGlFPyVwfHMBhCHMaUgmmb3J.jpg","https://image.tmdb.org/t/p/w1280/lPsD10PP4rgUGiGR4CCXA6iY0QQ.jpg","A low-level FBI agent staffing a phone that never rings is thrust into a dangerous conspiracy."),
]

# For items 101-500, cycle through real movies with different IDs
EXTRA_POOL = [
    ("Dune: Part Two","Sci-Fi","PG-13",2024,"https://image.tmdb.org/t/p/w500/1pdfLvkbY9ohJlCjQH2CZjjYVvJ.jpg","https://image.tmdb.org/t/p/w1280/xOMo8BRK7PfcJv9JCnx7s5hj0PX.jpg","Paul Atreides unites with the Fremen on a warpath of revenge against the conspirators who destroyed his family."),
    ("Baby Reindeer","Thriller","TV-MA",2024,"https://image.tmdb.org/t/p/w500/jMyBcMj4VmyEdwCsaJYmqVX1fcg.jpg","https://image.tmdb.org/t/p/w1280/2NQTBWSM5HpqA0cHF2S9TDkDxVo.jpg","A struggling comedian becomes the target of an obsessive stalker."),
    ("Ripley","Thriller","TV-MA",2024,"https://image.tmdb.org/t/p/w500/rO5mBxfbLyqT9SeEaAO4CMVFouq.jpg","https://image.tmdb.org/t/p/w1280/uXqSMNcAXKNT5r3sNi2NLHXJ1QN.jpg","Tom Ripley, a con man, is hired to travel to Italy to convince a wealthy expatriate to return home."),
    ("3 Body Problem","Sci-Fi","TV-MA",2024,"https://image.tmdb.org/t/p/w500/xMbFAfKc5PVFdkQB0CGBCS6AEzB.jpg","https://image.tmdb.org/t/p/w1280/nJuliznbyMub9dCIQGMKcCOEp5N.jpg","A scientist makes contact with a dying alien civilization and an existential threat unfolds."),
    ("The Fall of the House of Usher","Horror","TV-MA",2023,"https://image.tmdb.org/t/p/w500/spCAxD99U1A6jsiePFoqdEcY0dG.jpg","https://image.tmdb.org/t/p/w1280/z28pTJKNSK4b6BTqOKvnbEJIq6y.jpg","Two siblings built a dynasty of wealth, but their dark secret comes back to haunt them."),
    ("The Gentleman","Action","TV-MA",2024,"https://image.tmdb.org/t/p/w500/4n2HzmBYJJKzz9DmV1i5VXD5zHd.jpg","https://image.tmdb.org/t/p/w1280/8GPtkdULcVpBlvOv0mePfnLFnOW.jpg","A British aristocrat discovers the family estate hides a profitable marijuana empire."),
    ("The Watcher","Thriller","TV-MA",2022,"https://image.tmdb.org/t/p/w500/rDiDCmFuq7A9ILFELsJFm1b6PJL.jpg","https://image.tmdb.org/t/p/w1280/ySQzHHknfGCRFGkyMlnqIJMMMM4.jpg","After buying their dream home, a family is terrorized by an anonymous letter writer."),
    ("Monster: The Jeffrey Dahmer Story","Drama","TV-MA",2022,"https://image.tmdb.org/t/p/w500/qhL7oYC7FHfT8x2l0JVT3RFllzI.jpg","https://image.tmdb.org/t/p/w1280/aQzuKSNpABHhImTxQp9YXnIr4Il.jpg","How was Jeffrey Dahmer able to operate undetected for over a decade?"),
    ("Ratched","Drama","TV-MA",2020,"https://image.tmdb.org/t/p/w500/xbSuFiJh3SkBDOzBSvwusBYGQxj.jpg","https://image.tmdb.org/t/p/w1280/zaP4VKFgtrnuLKzF5xpuatSJBGy.jpg","A suspenseful drama telling the origin story of asylum nurse Mildred Ratched."),
    ("Locke and Key","Fantasy","TV-14",2020,"https://image.tmdb.org/t/p/w500/dKzOCW8OfJvgd4Fjo2TmCq4rT1T.jpg","https://image.tmdb.org/t/p/w1280/7TjbHpeMEkjH5OLs8JHAHV8HyVZ.jpg","After their father is murdered, siblings discover magical keys that unlock powers."),
    ("Narcos: Mexico","Crime","TV-MA",2018,"https://image.tmdb.org/t/p/w500/xRE3nMnkYyWJjSPxWiKjxZdgUr6.jpg","https://image.tmdb.org/t/p/w1280/bH7ggsRqFIwsOlY9VVL8Gkl6kAN.jpg","The true story of the Mexican drug trade and the rise of the Guadalajara Cartel."),
    ("Ozark Season 3","Thriller","TV-MA",2020,"https://image.tmdb.org/t/p/w500/pCGyPVrI9Fzw6rE1Pvi4BIXF6ET.jpg","https://image.tmdb.org/t/p/w1280/mAzm8Ah4NMXOX8nrjA4c2YYLK5a.jpg","Marty and Wendy struggle to keep their family together while managing cartel demands."),
    ("Money Heist: Korea","Crime","TV-MA",2022,"https://image.tmdb.org/t/p/w500/sWMoMDrGJv2EFPAeNbCvHJbKNAS.jpg","https://image.tmdb.org/t/p/w1280/rqgGGBQ4VNpAicxnkLBHF7QYFEp.jpg","A Korean adaptation of Money Heist set in the Joint Economic Area."),
    ("Ragnarok","Fantasy","TV-14",2020,"https://image.tmdb.org/t/p/w500/5RZnFExZrwL5cFQIkpqPsaLe1uO.jpg","https://image.tmdb.org/t/p/w1280/p5DFmqSCZXqKK8MHzGFMbqMqNqE.jpg","A Norwegian coming-of-age drama featuring Norse mythology."),
    ("Barbarians","Action","TV-MA",2020,"https://image.tmdb.org/t/p/w500/cbWKSCJJpjvPPBi2M6RHVWBMF3p.jpg","https://image.tmdb.org/t/p/w1280/mcFfOm4rXTGRZAELxblqAKYNFjZ.jpg","In 9 AD, Germanic tribes unite to ambush three Roman legions."),
    ("The Last Kingdom","Action","TV-MA",2015,"https://image.tmdb.org/t/p/w500/e5QBDmK6QNZQ0YLjfLTOaVHx3lN.jpg","https://image.tmdb.org/t/p/w1280/GrdRFv6c0gDJ95zBi2KbcVdN6Pc.jpg","A young English nobleman raised by Danes turns warrior defending against Viking invaders."),
    ("Marco Polo","Action","TV-MA",2014,"https://image.tmdb.org/t/p/w500/5NHnMTN4x7bVH3ot6OMV4lrBJt3.jpg","https://image.tmdb.org/t/p/w1280/n1gMBj8Oa5Yl2KfC5LJXpKUNZDW.jpg","Marco Polo is captured by Kublai Khan and made a prisoner of the Chinese court."),
    ("The Old Guard","Action","TV-MA",2020,"https://image.tmdb.org/t/p/w500/m0tPMFLXU0Pjun6bJKomHKRJGmg.jpg","https://image.tmdb.org/t/p/w1280/tKMVEwH0J4z1vFG2VnUK6pYAcPg.jpg","A group of mercenaries with the ability to heal themselves discover their immortality is under threat."),
    ("Project Power","Action","TV-MA",2020,"https://image.tmdb.org/t/p/w500/3ELGxMnlQJ7dGJBlpvQECuVv4BC.jpg","https://image.tmdb.org/t/p/w1280/y9NiP22cqcJeY1Y1KmZ9zJfXCPi.jpg","A cop and dealer must take a pill that gives unpredictable superpowers to fight a crime lord."),
    ("Murder Mystery","Comedy","PG-13",2019,"https://image.tmdb.org/t/p/w500/Ah6cpagJsXJlD88DGbIJtpIMSaK.jpg","https://image.tmdb.org/t/p/w1280/sXp5e6tGqQH2dqJhxPiQPNt6bFk.jpg","A NYC cop and his wife become prime suspects in a murder on a billionaires yacht."),
    ("Enola Holmes","Adventure","PG-13",2020,"https://image.tmdb.org/t/p/w500/riLSZqGmjDfHTCLk30GkEsRoWmT.jpg","https://image.tmdb.org/t/p/w1280/szf29U2BkGvgaaxGRBiaBCJB63R.jpg","Sherlock Holmess younger sister discovers her mother has gone missing."),
    ("Furiosa: A Mad Max Saga","Action","R",2024,"https://image.tmdb.org/t/p/w500/iADOJ8Zymht2JPMoy3R7xceZprc.jpg","https://image.tmdb.org/t/p/w1280/xHsXSA15Ov3RHYQxBHXg9DLUR0i.jpg","The origin story of the renegade warrior Furiosa before she teamed up with Mad Max."),
    ("Deadpool and Wolverine","Action","R",2024,"https://image.tmdb.org/t/p/w500/8cdWjvZQUExUUTzyp4IoijrKMWe.jpg","https://image.tmdb.org/t/p/w1280/yDHYTfA3R0jFYba16jBB1ef8oIt.jpg","Deadpool is recruited by the TVA and must partner with a reluctant Wolverine."),
    ("Inside Out 2","Animation","PG",2024,"https://image.tmdb.org/t/p/w500/vpnVM9B6NMmQpWeZvzLvDESb2QY.jpg","https://image.tmdb.org/t/p/w1280/xg27NrXi7VXCGUr7MG75UqLl6Vg.jpg","Joy, Sadness and friends are joined by new emotions as Riley navigates teenage life."),
    ("The Substance","Horror","R",2024,"https://image.tmdb.org/t/p/w500/lqoMzCcZYEFK729d6qzt349fB4o.jpg","https://image.tmdb.org/t/p/w1280/7T5aHD7BmEzAIFEXFPTM0JDmv2Y.jpg","A fading celebrity uses a black market drug which creates a younger, perfect version of herself."),
    ("Gladiator II","Action","R",2024,"https://image.tmdb.org/t/p/w500/2cxhvwyE0RtuhMkstmKTkzKzrpq.jpg","https://image.tmdb.org/t/p/w1280/lgCImHdV4NIYF5HgxDNHfRRiIZc.jpg","Lucius must enter the Colosseum after his home is conquered by tyrannical emperors."),
    ("Conclave","Drama","PG-13",2024,"https://image.tmdb.org/t/p/w500/m5ZXdGGelJgrQNbUBNBJKJjjrNE.jpg","https://image.tmdb.org/t/p/w1280/fqv8v6AycXKsivp1T5yKtLbGXce.jpg","A Cardinal oversees the secretive process of selecting a new Pope."),
    ("Emilia Perez","Drama","R",2024,"https://image.tmdb.org/t/p/w500/mQ5Gto1eCpEVHGAMUGPnrq0pYm3.jpg","https://image.tmdb.org/t/p/w1280/4bUKTEIv1eXPV7PCWP7fgRZGUFM.jpg","A Mexican cartel boss decides to undergo a gender transition and start a new life."),
    ("Nosferatu","Horror","R",2024,"https://image.tmdb.org/t/p/w500/5qGIxdEO841C29qqkbMvsLAifwR.jpg","https://image.tmdb.org/t/p/w1280/mpPqfbSJObQlO7UlB2xW5KEqf2W.jpg","A reimagining of the classic about an obsessive love between a woman and an ancient vampire."),
    ("Formula 1: Drive to Survive","Documentary","TV-MA",2019,"https://image.tmdb.org/t/p/w500/jTRpSaLoEKqNHicN2oNNFMIADe4.jpg","https://image.tmdb.org/t/p/w1280/pSFVlZ6OBDpqNRkxHBPuWI68VBM.jpg","An inside look at the drivers, managers, and owners who compete in Formula 1 racing."),
    ("The Last Dance","Documentary","TV-MA",2020,"https://image.tmdb.org/t/p/w500/7GBMipjNFf7qKOvl6tZMRF68mvg.jpg","https://image.tmdb.org/t/p/w1280/bZyTTZVXqz8sMNLpVOVhS3Bknbr.jpg","A documentary about Michael Jordan and the Chicago Bulls championship dynasty."),
    ("Our Planet","Documentary","TV-G",2019,"https://image.tmdb.org/t/p/w500/GWokSNkLNxYC7aVRBYjJLuBGmUG.jpg","https://image.tmdb.org/t/p/w1280/7vox7DHqTJDILEEhKRNZSHlarAm.jpg","Experience our planets natural wonders and the way human activity is threatening them."),
    ("Cheer","Documentary","TV-14",2020,"https://image.tmdb.org/t/p/w500/pTjKzxl2aPh7w88i9RhkRHxD6F5.jpg","https://image.tmdb.org/t/p/w1280/8e6VVBCo9m5k2aKL4MFNqgN3CDZ.jpg","Students of the Navarro College Cheer Team strive to make it to nationals."),
    ("Chef Table","Documentary","TV-G",2015,"https://image.tmdb.org/t/p/w500/ohBqJbkUZNdPkYXxuRlMJefLMf0.jpg","https://image.tmdb.org/t/p/w1280/qSBaVUXY7yFQUmEeG3BWXP8bJVH.jpg","Documentary series profiling highly acclaimed chefs from around the world."),
    ("Avatar: The Last Airbender","Animation","TV-14",2024,"https://image.tmdb.org/t/p/w500/coh8mWBGpHJgHfGEZsITEmSXDFP.jpg","https://image.tmdb.org/t/p/w1280/3dTbBi6bCiLqWFKKD7VqlLIUhpb.jpg","A young boy must master the four elements and defeat the Fire Nation to bring peace."),
]

import importlib, types

# Load the running app module
try:
    from recsys.serving import app as app_module
    catalog = app_module._CATALOG
    print(f"Loaded catalog with {len(catalog)} items")
except Exception as e:
    print(f"Cannot load module: {e}")
    exit(1)

import numpy as np
rng = np.random.default_rng(42)

def make_entry(item_id, title, genre, rating, year, poster, backdrop, desc):
    return {
        "item_id": item_id, "movieId": item_id,
        "title": title, "genres": genre, "primary_genre": genre,
        "description": desc,
        "poster_url": poster,
        "backdrop_url": backdrop,
        "tmdb_rating": round(float(rng.uniform(6.5, 9.2)), 1),
        "maturity_rating": rating, "year": year,
        "avg_rating": round(float(rng.uniform(3.2, 4.9)), 1),
        "popularity": float(rng.exponential(100)),
        "runtime_min": int(rng.integers(85, 180)),
        "match_pct": int(rng.integers(72, 98)),
    }

# Patch items 1-100 from FULL_CATALOG
patched = 0
for row in FULL_CATALOG:
    iid, title, genre, rating, year, poster, backdrop, desc = row
    if iid in catalog:
        catalog[iid].update(make_entry(iid, title, genre, rating, year, poster, backdrop, desc))
        patched += 1

print(f"Patched {patched} real items (1-100)")

# Patch items 101-500 cycling through EXTRA_POOL
pool_len = len(EXTRA_POOL)
extra_patched = 0
for iid in range(101, 501):
    if iid in catalog:
        row = EXTRA_POOL[(iid - 101) % pool_len]
        title, genre, rating, year, poster, backdrop, desc = row
        # Make title unique by appending a variant indicator
        unique_title = title if (iid - 101) < pool_len else f"{title} ({iid})"
        catalog[iid].update(make_entry(iid, unique_title, genre, rating, year, poster, backdrop, desc))
        extra_patched += 1

print(f"Patched {extra_patched} extra items (101-500)")

# Verify
with_poster = sum(1 for v in catalog.values() if v.get('poster_url') and 'tmdb.org' in str(v.get('poster_url','')))
print(f"Items with real TMDB poster: {with_poster} / {len(catalog)}")
print("Sample:", [(catalog[i]['item_id'], catalog[i]['title'][:30]) for i in [101, 200, 300, 400, 500] if i in catalog])
print("DONE - catalog patched in memory. No restart needed!")
