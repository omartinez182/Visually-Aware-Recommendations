# Copyright 2018 The Cornac Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ============================================================================
"""
This data is built based on the Amazon datasets provided by Julian McAuley @ http://jmcauley.ucsd.edu/data/amazon/.
We make sure all items having three types of auxiliary data: text, image, and context (items appearing together).
"""

from typing import List

import numpy as np

from ..utils import cache
from ..data import Reader
from ..data.reader import read_text


def load_feedback(reader: Reader = None) -> List:
    """Load the user-item ratings, scale: [1,5]

    Parameters
    ----------
    reader: `obj:cornac.data.Reader`, default: None
        Reader object used to read the data.

    Returns
    -------
    data: array-like
        Data in the form of a list of tuples (user, item, rating).
    """
    fpath = cache(url='https://dl.boxcloud.com/d/1/b1!ACQqSpGVIZsOBeZhv_RHYt-gt8cvbeWTE8SN9DS6Ji3BRUfm9KJN6U2ncjj0CTuz05u-cR6g4Pad3ubtU6FJXQ6zPJdvpBTISDnb4ljl6PuF-jELh9iU6N33kgDa06EPRIJCn4VFIFDm2NkpZCsuBFkt8O3HV2AAfOnyJ72O7N3yCS2H7nR2QZJ8ClONVxaNSGfGZ6zf5O-4kIMZ-P2PydNnX8dZ0FLi7pbwU_tw94aiGJdGq84E0f24CuUs_ZdLUm6JMWNJwXz15-rfwGVQXPsDcXOy_11-ygYxJWOpLu8SFYcXQ0PYc4yyt4wqQfDSAuqTdHdVZtI6sMjnSeKTGhwGX_Od2Erbk0z64K6oI53UIsRWChSW9tc4nlmixFkw-0tuRmC5DayiPyLo60sd7zfaa5b-IrjeuJhmcfZPQfpdjtwEd2MfqWex-IXOAWptgtZlWdsc7TMR2h_SHS1So48oNlQ7P-KxKAO_4Wm2rtVGnifFF_VDkFaUJSrfe-M8t-bRgYv1KQxxI6lIaFosMpkIjws3ke4_gqwoQXUXboQr3DW-VGJkyD4Y1IagymYvfvekfteovr5QeRW8pUprnLIkoBr6qUQ5zGpuKmw27akCWcq7y6kbCYzjXlDoav9_98UGVjgKOJasKMYtQrzm6xlgwtpU0MXOhkE2p7jMNiToqIFP1CHcE31a33oqdUBfE-Ot4elu0ae3UdOm1q8Gadsh8Fmz4CQgFuqb_0OwrBIl7puv8M48vjRc9nOUEShbwokKneSTe5QKk09poiQOjOoqlvTsZZskstcGEjFQAxHB6Fyx6kgqIzrdrkAZgKkSFePGWX6thrU4jA49ekzWwe6leY5mnplIN-P5DNGW8iLIUqo_RkpeoZvngneC9_bXCsO5Zn0C2s9X-M6bmP-5kHBZTZ2hRkD2-L58K9Q23qwWEpEmvy7zQUD7nij9SgjJdGBXxtl2ilZUcdzUJYdTcQwEWrMErGpKt4mtlgatNNRkSFFVs9zC7vF9EWraeiIx_q3B7_lI-K-K6WK_ZMwe0O-GKDE_dM5vHI9Suy2aSN9pfm6e18H3MmFJiqlk8t9uuwajev_HrqfrLAvnET8tRlHbLEQn_44CytNCaPw-AEyj8dC7AeZf30gFj6Lx_-IubDeGXEjnpj2e7DL9DNxa2-vuH9Sn2U8qoDpqZQUqMbYkxzwiF-s_214Ce-Pxp_tkG9K5Og0ZOVsUZkCp_chdGsSORUkwfxrsyPKT2hkOug7re7caibaiHIzpvLULtKNJ6ufwtAykd10XvKIR0iT3gLgErNVe7Z0JOaPBi4BG0f3TTVDJFEkxh4EcjgaXSEWGiD5jSG-MmKlN8LDzpNHda9BpqNL_UhlB7LavS-BrXZP1JjpZUTgzBNpXxbQbmAO7AUMf1EJKD_z5Ee2Qika2rHg./download',
                  unzip=True, relative_path='amazon_clothing/rating.txt')
    reader = Reader() if reader is None else reader
    return reader.read(fpath, sep='\t')


def load_text():
    """Load the item text descriptions

    Returns
    -------
    texts: List
        List of text documents, one per item.

    ids: List
        List of item ids aligned with indices in `texts`.
    """
    fpath = cache(url='https://dl.boxcloud.com/d/1/b1!Ehv-67Z8oXiXdn9z9hZQNwWjx-J-1UXdDiHLK4rtWGfm_1OAXu4sLgSfjtZXLf0RNUM2sYX1S2eXRtVs08a_u5AlWv8TALbpdAKWsT65wWrPPGlKMWvMI8qkYUJypCQ2v8NaRVAbOHwKJZ2oNGITwYX5xdiUujPrJvz2oL7qV9CLBtt5R3kBdm_usct-VHR_QI5XnjV7fs5Ky2Lwvt00empugSwhtuiFi_p-eOls2NklMx3urwo5uokSLxVtQbu2S9_MaNNqs3R8NBkA7F0NWDwxajDlVF7nZLWAlKv5_gOzZY10IYMg6Mh4UQ33h-RIPhLpD1ZkyQdIfE-u3tGoNQSwMb2xvd_PCIB3geL9_6j95gpaoBXEr-4FX-sUKnTG8dE7QsjpXUYhVtxrWH_cxSW7vKnoN0uRkIrAqZnwMe2OAr6V-sSosaCrOegRmbWBnF2T7Uy501BvkqxUG-JpeUSjEtg4xR6_QVDTi21i-oZr7WPxJCDNIZarDmBsVTUo1PDDckIgfimQAfWk1c-hUSuQXj4TqJ1g4yn2uCMF-LanDqqGsvANxu778Cu7zMAPrw1AEqDLEB6FjOWjkCZu1H-cWkRuKbLlZ61FhysnAB9qsbW6JpE7go7409ThoAUG_6mVAuyk0fjbgjFdqLGba454paIW7oOUhBbaB00PdFEuOJ7g54oDynmUs7Jm1aQOKiN3h8nFrhN0dq2peJsG8UqSksuN0Hk-rP2WAnBTwIWGVjLsi6EkK0SqF2ZbAMVCnorfrY5OXtFYvMp-lLO5USM4oCxuRmgGOgONXByORtmRClcUE3ey_WbE9SgTiJbrLRppL7GV6FqmOORJ-JQSivKIXEP3Kcyt0zvbYeIuxr7qYzbgaJ1MTbTCSW-FFPFZ_byLZcI-VmnftjQsuhu7zTDnfupXQJrdRCaohRBHfzeDvwzTRVeQFzyTH-IuaKFexkg-YLIdLbHpwt7GxROmpPv9JJxuAt9EerwQx2mmlhV5494izzhYRaGPdh_EOatWvH_m_i-stUVz-WqzI1BDNaLdsNCUGVsruByq4i27tPHOIeGIj3VgDsGXfQ3AmdAeYOFYAMMfWTrWZA8QwBkiOkXiW2YT7D2HyEZVGNptS3qWTjDSYJdMAr2sd_AqcfXrJUgkonB0dbjilRJXtBru-CIlzd-YB7M7qkaSvmJFzRc8fZGzcpUU-kMX5SVqvCl0RjPqBqtZI7yFf34-cYh-HVbY3AUNJ2wHLjPZN55j4TvWMPVLwDSH3uZ4qnzIclgZ_mM9gTkWsfISuirtH0vDX5Tf3gqPzBJpPADdM8Ps9NM8dCV73VLMytfpslv3OIZHTmd8mXs7yTUQX2F0e9xvn946hV5a4eRlTdtrkGAe-mfDf4CJiMELnoz4H2UyqCFwOoOguD_V4uOdt5lVUDso/download',
                  unzip=True, relative_path='amazon_clothing/text.txt')
    texts, ids = read_text(fpath, sep='::')
    return texts, ids


def load_visual_feature():
    """Load item visual features (extracted from pre-trained CNN)

    Returns
    -------
    features: numpy.ndarray
        Feature matrix with shape (n, 4096) with n is the number of items.

    item_ids: List
        List of item ids aligned with indices in `features`.
    """
    features = np.load(cache(url='https://dl.boxcloud.com/d/1/b1!bxbafPOZVQM1o_4roV0IH_qE2Oyy4wLwgEBvBqBlXm12G5C4EFfaEBOrIvKOl2flwWPh8QzOkjrL9LLNuAk8qp8l8WkuysJpsHctsWNfXw5okvt5I8DtOrrX_zp0NRGzVfsyQzWhLuurQf68N3U-mXgp_v6Alu2aqcDn0QomQB-spH3t44SfU4aCNXbbufDx8dI4XeM1XEXx5twICuTHKNitxhWuOftNARX1n7GVEXJzsDLhRRUOWtp6r1ALqJLGEr2PD-gUENmHPoqF72AbCCf9TmjB7naEDWtxMpn_xNwqwbuNKjyhBrqSjFijPufGH8BGA2MBiOeeGo_UD_5H6oQ_0doXoQdDA65isSrHkYwb5FHk8ctCS2fiC1QV_TIuFk3CE-Bp9TwW70SFKIvF8jYAhFDgPy2C9dAHqNQ18vNCISIq51FBwm9ybFzrlVnaFCXR-WbXWLAKP_Hw3m3hnhh5C-X-x15ffngccAk8B5y6M3CzuoDstUYok5fts6BG6ZSUtwWq8tzZWZTm8V-zjD4LEROytBwZf1kIy67pT3ASW6GaMAX6lMTZUr7I368VnMI9EJAA7b7OU-F1g_kNWgBw3hR-ZrOgUPmMMMuviBBDZ1cS7qstG6XDMlWVrJYmz2I8v0Oi71eM6KF0XApmGILr0q_it2YIB_hVQ1RMga9AAhawKKGx1Dne85kfQ0nNOAY4T9ADDWQekPwYkSUYKSxyO17QDJsNPB0Hoo0MueyQixhPvTIWIxvzwRDU0inkJ1tWdIsE-aCa5EKWDziFoRHrvh8FmynGGNk0WAfzwjbHoTAd59SkFi0OdjEaMzhXGTep-VQKnSuKoPouQzboWhPGyL3DJMjb09kICibst-Ku34D5YH76_en4RK2fbhra3XP9FEM8296GWSwoy_yGk1G69Rd90h3D38PRdzyza4ZjUqcIZ-bKbokUqYEgMjZpEEDmFZCFMY-xp8Jt0m-b-qyvUN6YtLDMmXKs52a9HOYBDHo8rlS1sa841Ny9t5tNTbMnOVpiHm4V0cLOcg4fq1qqrW08k9SQd3tLeTAsTEDi7fcL3y2EKpeLV14KDnAolDMqGcvSaxZyylGxrvCxntbvAZcf4LGmAR7DuGJ0_vHQDTXYPM6vh85Td-vH_t8JO0zPsEBKAfSQSRFMZ9v5yUrctzkfQYLE9sGSp6HwTdkqqKboOPGAZEbt8gne_3WYaNMzgRRIODJnNqsFcnMoIt0R1DqoWoQjsDYTyPVxLTG6V5yk6kIuJ70ZNHHo7WXPiKPspQKECBaI6y6EktCPR8C2s5djrdQfmUggkG4f0KOhp41k8RnQUijMKGmBusHnI0HAvKX-pIitTz0uAjmKhbNP0aAlSrpO_Y0fRN7g0ZlyPmmeCjkqQwCpjzJOHn14-aZo2DO2P3N_bAFRpmda1EAmVaMdqnzeM_a_Lfq6hTL-tb2B-gyCd7o1B3xbweJIHxqVub1c/download',
                             unzip=True, relative_path='amazon_clothing/image_features.npy'))
    item_ids = read_text(cache(url='https://dl.boxcloud.com/d/1/b1!YygJabrPpubrI06FiijvyKhHgwUPjl7S4HI4M-yQVGrlrkPYH77v8wwNrVtT0z7wtW9-CTfB6EGZotUgNkyVQ7EgbQcel19xhHaGVFspH5enUJbjilMdnhGMWJKSGAYh3xFCsk9VEj9aEfQed0im1uRy8uwMjSPgogHPgWHUwaC4BYBYtKUxtAztpTK9NLBUoCdjOs7zvzaqM0pDf8uEAZDMCl_TTt82j7hMbwWHRByxEfNumP7wbKUYnXpZowJF5X21V5tO6lgFrcQ3V654NPWFELyOYFUL1wzUa9mMOI2NH1d4x87c-9bLV4eqKSLMRQuWYjvLMgLmdyJDI4weSAxU9lXsUNxeOGkctFGu1hi8d7ecs5hJKD-0IKjUoT4tSO-Qh9Uh18blsipIC9HXeWEBKZmaVmWmReaz_IuDrVpjrI72zWuBZMjXoMmB_lD7sVn40pai2IregW5_2hHVqXpot-70mx7YLwcsZODaEybjkZLHXrb5apMrNNEDYEPoVxxAUC_UojQXiEYmCsOkxNAkfplsj87gio2IeQdaBMcanDVO8b6NwAXXoXFAcJK3env1XsQeM7bnGvo2q3yTOn-N_f7XXLmzkrjKF3yIuN0nfeeHNmk7Lhlvg78hZLVklFTP6VAhLsEh7a7U3IMqZ8BkIIhra6vFF4oAIQ1yUmZfRdQrnv-KFwRu10_gaUMD5pp62iHd7TgbXFEjCIMecOPVWDeLHeEgLAIYW551wjPmkew5RmM06bAe5ZbEn1b6TsmFoJAuIFJ9F_bKxxYWOtOdGX0jMoA4YrSm7Ms0VEfhDrJv569L75XhFuPCIIVRIk_gA7nhrAHWnEG9CPPGXGp68tLPke8aGLpvoK4YCFGdZFLG9NPqeH5y7tRIL0x3qMlgerRahPeqH9rJqpksECEH8TiBstiM4Zw9AORLvfkjqxCp1cQK1c925gIPUWVYBB3jLfZnnIzg3DHw7SmQNDywYljoTXi9QmELpXXE9tA1KTGnorcodXbKpQALqzjt0CkmUnAvCicrpCHVlZZYPjQtHprIwB24WzPBszFwiCregCfSuBB2AGyI6mokpJ1y_lVKIWNMF_cHfMnBPTOtuiJU_kIWG_u3bSOznPy_WLCIud6SYfT_LdhReshyZLtzdTCk4iboVw1ETqxfEVdYIMoUbsJ97ampbQ3tedhaiaZUVvHr2n98vry-aGgiU-Pgc3PfSiHJD3ef1z-7tMSuO9mdRcbwzEhm10kApVOpVE_deSot3Cm9Scf-nWH5vZ3nuNoayCuGYL9WaOREBulQlnxFqGU4JgAh0tVOtkPhnSG_o-5dCWBsvA4pBC-QqYk3ZOx2dpzFFjOOe0jPH9A61l2oAvstDHDthEVBHxe3H19EDK49akW2IsuonL8_uDUDDh-OwjJKDSUFoa5NCgrkbGgwHLd_bG1lasujVxh8MTlWPX7_DZ8EftLwwJDa4xu31ZveKrxD/download',
                               unzip=True, relative_path='amazon_clothing/item_ids.txt'))
    return features, item_ids


def load_graph(reader: Reader = None) -> List:
    """Load the item-item interactions (symmetric network), built from the Amazon Also-Viewed information

    Parameters
    ----------
    reader: `obj:cornac.data.Reader`, default: None
        Reader object used to read the data.

    Returns
    -------
    data: array-like
        Data in the form of a list of tuples (item, item, 1).
    """
    fpath = cache(url='https://dl.boxcloud.com/d/1/b1!nIPopYq-OrLL1EOOpBa2JorzJH7ZNZstyREplLFZLjobxKsVKiS9OI6JLKf7gFiIr2gIpztvEv-QyDa87iaCF62Mw3cGq7Ipo7cmktnFfTQuMurf5ngnaHX1TPn-MevmFd4Yx6V9lQ2mYD6OJlMq42utAgWslpaqptiD1dG-RPuXiVwdc2vGNKITbq1esAy5BaevM8qxI2O3npsGRPZkxBdeknl7X9Ic7v2vWBjB3vZeV1HFo-FoJI-fYrLUMkJY2uCGMbq1JbVffMJF0hsXW_W5bkS6Y670X8N9Ngqf3PQhQ4UE3v6oQDnkUA6Dk9Mdz2iwqwa_yA6uNJ9Osl2LxwQdDud49FMxEAo3sFm9Hrh_v4BZHtnP47qX_eDSQaCxxY2YutHR92IYsFd5bJmXyGw_23Y8bVG9g2jo-OfNIKRaWFQ5EeA80l6ZqD2ojNqVOvdMXjphzyZ1A8BHW0QsHDz9ehW2jnASyx_QZ30hEWucbU9P34GR-xANGbpvpkWc5DYYbghJDUZbVKoldLTQX0rgJ0A5j7_Y0RbVK4T2bnQH5iRgVWO6qyS3fxDQEpeg5Ypm86UJAoIx96Ec44X0imtVK-KrBIJ8_dNfKne2onkttQm1LDJSKcSPNU-qjVZsF7He1kk4IaP_LqKRNMs4yjzegAUBZG2X3HcDVr5aTJmXFs25ZVfEyIvLiXBi0PvnFeRQYgb5LgfreMNU1egi5A2gMUy5m1wL37onuzhnaTSk4yXMqkpMS1LyA5N-hiuQwG1aizzzqpA-6tdS1XQ6ShJAmCTD76JzU9mwCgV-CJ7vEEPWYfksHpQnFAD1_wHUuVKaz8uuFbdtYdXvgiXatsFcxRSOtu4VcC42CgAaT2lmfwTYp6KMGGv2dtbcD_SHV2eFrtHOvZ9DLKfhJOijsqTfAzLuUDqnPX7hzgwzUx0-0vWOL7VK6RgOFp1qcGdyuf8j3wLOK4D6aqzE-l_JgDtPShtLf4mCmgCVvNJ2yg7pvbiuyAZfK5BTi2OYBQvIMMj-Xr9eoXh73_zWE6jRGRlSEPpKwKTBDwmI3lKX-cMdaC_xBoyu7sYiH6F8eObVMuHiaI9gf6Ln5srF_wAvgPZAddkaKbgbF1YYb_Emc2EHnT8LOctL5CTb0xoSx8DpnMuuKoniXKfxOVEDOx4-nH6MWHFD1-tWMHb3_XLkmkUZGcH01BgpjW5DlOPaGfgWnwVuSRDHH7TBKooyVqfIAyHgK9KaQMx72RtY3VKMLtedE7M2y6oY7AsjgP--ekh1Oy6Tsd2pn4-P0RCinA1-TcMy9AYrmC1_b9-EmiBzGqFmMI6CrYBQMuIpqpz7MKy6GufQrqqMjAXuqhnArGCugRYQcwOCA83nRH5GVcVU9z9I_MY4r9Lh/download',
                  unzip=True, relative_path='amazon_clothing/context.txt')
    reader = Reader() if reader is None else reader
    return reader.read(fpath, fmt='UI', sep='\t')
