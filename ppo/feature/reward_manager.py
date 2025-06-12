#!/usr/bin/env python3
# -*- coding: UTF-8 -*-
###########################################################################
# Copyright © 1998 - 2024 Tencent. All Rights Reserved.
###########################################################################
"""
Author: Tencent AI Arena Authors
"""

import math
from ppo.config import GameConfig

# Used to record various reward information
# 用于记录各个奖励信息
class RewardStruct:
    def __init__(self, m_weight=0.0):
        self.cur_frame_value = 0.0
        self.last_frame_value = 0.0
        self.value = 0.0
        self.weight = m_weight
        self.min_value = -1
        self.is_first_arrive_center = True


# Used to initialize various reward information
# 用于初始化各个奖励信息
def init_calc_frame_map():
    calc_frame_map = {}
    for key, weight in GameConfig.REWARD_WEIGHT_DICT.items():
        calc_frame_map[key] = RewardStruct(weight)
    return calc_frame_map


class GameRewardManager:
    def __init__(self, main_hero_runtime_id):
        self.main_hero_player_id = main_hero_runtime_id
        self.main_hero_camp = -1
        self.main_hero_hp = -1
        self.main_hero_organ_hp = -1
        self.m_reward_value = {}
        self.m_last_frame_no = -1
        self.m_cur_calc_frame_map = init_calc_frame_map()
        self.m_main_calc_frame_map = init_calc_frame_map()
        self.m_enemy_calc_frame_map = init_calc_frame_map()
        self.m_init_calc_frame_map = {}
        self.time_scale_arg = GameConfig.TIME_SCALE_ARG
        self.m_main_hero_config_id = -1
        self.m_each_level_max_exp = {}
        self.reward_groups = GameConfig.REWARD_GROUPS

    # Used to initialize the maximum experience value for each agent level
    # 用于初始化智能体各个等级的最大经验值
    def init_max_exp_of_each_hero(self):
        self.m_each_level_max_exp.clear()
        self.m_each_level_max_exp[1] = 160
        self.m_each_level_max_exp[2] = 298
        self.m_each_level_max_exp[3] = 446
        self.m_each_level_max_exp[4] = 524
        self.m_each_level_max_exp[5] = 613
        self.m_each_level_max_exp[6] = 713
        self.m_each_level_max_exp[7] = 825
        self.m_each_level_max_exp[8] = 950
        self.m_each_level_max_exp[9] = 1088
        self.m_each_level_max_exp[10] = 1240
        self.m_each_level_max_exp[11] = 1406
        self.m_each_level_max_exp[12] = 1585
        self.m_each_level_max_exp[13] = 1778
        self.m_each_level_max_exp[14] = 1984

    def result(self, frame_data):
        self.init_max_exp_of_each_hero()
        self.frame_data_process(frame_data)
        self.get_reward(frame_data, self.m_reward_value)

        frame_no = frame_data["frameNo"]
        # if self.time_scale_arg > 0:
        #     for key in self.m_reward_value:
        #         self.m_reward_value[key] *= math.pow(0.6, 1.0 * frame_no / self.time_scale_arg)
        
# REWARD_WEIGHT_DICT = {
#     "hp_point": 2.0,
#     "tower_hp_point": 5.0,
#     "money": 0.006,
#     "exp": 0.006,
#     "ep_rate": 0.75,
#     "death": -1.0,
#     "kill": -0.6,
#     "last_hit": 0.5,
#     "forward": 0.01,
#     "total_damage": 0.1,
#     "hero_hurt": -0.1,
#     "hero_damage": 0.30,
#     "no_ops": -0.001,
#     "in_grass": 0.001,
# }

        # 奖励分阶段（前期注重发育，后期注重推塔，KDA）
        # 1820步为一分钟
        if frame_no <= 5000:
            self.m_reward_value["money"] *= 1.2
            self.m_reward_value["exp"] *= 1.2
        elif frame_no <= 10000:
            self.m_reward_value["tower_hp_point"] *= 1.2
            self.m_reward_value["kill"] *= 1.2
            self.m_reward_value["death"] *= 1.2
            self.m_reward_value["hp_point"] *= 1.2
        else:
            self.m_reward_value["tower_hp_point"] *= 1.5
            self.m_reward_value["kill"] *= 1.5
            self.m_reward_value["death"] *= 1.5
            self.m_reward_value["money"] *= 0.8
            self.m_reward_value["exp"] *= 0.8

        # 局内奖励随时间衰减
        if self.time_scale_arg > 0:
            no_decay_keys = {"hp_point", "tower_hp_point", "kill", "death"}
            decay_factor = math.pow(0.6, frame_no / self.time_scale_arg)
            for key in self.m_reward_value:
                if key not in no_decay_keys:
                    self.m_reward_value[key] *= decay_factor

        # 计算reward_sum
        tmp_sum = 0
        for key in self.m_reward_value:
            tmp_sum += self.m_reward_value[key]
        self.m_reward_value["reward_sum"] = tmp_sum

        # 奖励分组求和返回
        reward_group_sum = {}
        for group, keys in self.reward_groups.items():
            reward_group_sum[group] = sum(self.m_reward_value.get(k, 0.0) for k in keys)
        self.m_reward_value["reward_group_sum"] = reward_group_sum

        # # 打印分组奖励信息
        # print(f"\033[94mFrame No: {frame_no}, Reward Group Sums:\033[0m")
        # for group, value in reward_group_sum.items():
        #     print(f"{group}: {value:.4f}")

        # # 打印奖励信息
        # print(f"\033[92mFrame No: {frame_no}, Reward Values:\033[0m")
        # for key, value in self.m_reward_value.items():
        #     print(f"{key}: {value:.4f}")


        return self.m_reward_value

    # Calculate the value of each reward item in each frame
    # 计算每帧的每个奖励子项的信息
    def set_cur_calc_frame_vec(self, cul_calc_frame_map, frame_data, camp):

        # Get both agents
        # 获取双方智能体
        main_hero, enemy_hero = None, None
        hero_list = frame_data["hero_states"]
        for hero in hero_list:
            hero_camp = hero["actor_state"]["camp"]
            if hero_camp == camp:
                main_hero = hero
            else:
                enemy_hero = hero
        main_hero_hp = main_hero["actor_state"]["hp"]
        main_hero_max_hp = main_hero["actor_state"]["max_hp"]
        main_hero_ep = main_hero["actor_state"]["values"]["ep"]
        main_hero_max_ep = main_hero["actor_state"]["values"]["max_ep"]

        # Get both defense towers
        # 获取双方防御塔
        main_tower, main_spring, enemy_tower, enemy_spring = None, None, None, None
        npc_list = frame_data["npc_states"]
        for organ in npc_list:
            organ_camp = organ["camp"]
            organ_subtype = organ["sub_type"]
            if organ_camp == camp:
                if organ_subtype == "ACTOR_SUB_TOWER":  # 21 is ACTOR_SUB_TOWER, normal tower
                    main_tower = organ
                elif organ_subtype == "ACTOR_SUB_CRYSTAL":  # 24 is ACTOR_SUB_CRYSTAL, base crystal
                    main_spring = organ
            else:
                if organ_subtype == "ACTOR_SUB_TOWER":  # 21 is ACTOR_SUB_TOWER, normal tower
                    enemy_tower = organ
                elif organ_subtype == "ACTOR_SUB_CRYSTAL":  # 24 is ACTOR_SUB_CRYSTAL, base crystal
                    enemy_spring = organ

        for reward_name, reward_struct in cul_calc_frame_map.items():
            reward_struct.last_frame_value = reward_struct.cur_frame_value
            # Money
            # 金钱
            if reward_name == "money":
                reward_struct.cur_frame_value = main_hero["moneyCnt"]
            # Health points
            # 生命值
            elif reward_name == "hp_point":
                reward_struct.cur_frame_value = math.sqrt(math.sqrt(1.0 * main_hero_hp / main_hero_max_hp))
            # Energy points
            # 法力值
            elif reward_name == "ep_rate":
                if main_hero_max_ep == 0 or main_hero_hp <= 0:
                    reward_struct.cur_frame_value = 0
                else:
                    reward_struct.cur_frame_value = main_hero_ep / float(main_hero_max_ep)
            # Kills
            # 击杀
            elif reward_name == "kill":
                reward_struct.cur_frame_value = main_hero["killCnt"]
            # Deaths
            # 死亡
            elif reward_name == "death":
                reward_struct.cur_frame_value = main_hero["deadCnt"]
            # Tower health points
            # 塔血量
            elif reward_name == "tower_hp_point":
                reward_struct.cur_frame_value = 1.0 * main_tower["hp"] / main_tower["max_hp"]
            # Last hit
            # 补刀
            elif reward_name == "last_hit":
                reward_struct.cur_frame_value = 0.0
                frame_action = frame_data["frame_action"]
                if "dead_action" in frame_action:
                    dead_actions = frame_action["dead_action"]
                    for dead_action in dead_actions:
                        if (
                            dead_action["killer"]["runtime_id"] == main_hero["actor_state"]["runtime_id"]
                            and dead_action["death"]["sub_type"] == "ACTOR_SUB_SOLDIER"
                        ):
                            reward_struct.cur_frame_value += 1.0
                        elif (
                            dead_action["killer"]["runtime_id"] == enemy_hero["actor_state"]["runtime_id"]
                            and dead_action["death"]["sub_type"] == "ACTOR_SUB_SOLDIER"
                        ):
                            reward_struct.cur_frame_value -= 1.0
            # Experience points
            # 经验值
            elif reward_name == "exp":
                reward_struct.cur_frame_value = self.calculate_exp_sum(main_hero)
            # Forward
            # 前进
            elif reward_name == "forward":
                reward_struct.cur_frame_value = self.calculate_forward(main_hero, main_tower, enemy_tower)
            # 对英雄伤害输出，防止立正挨打
            elif reward_name == "hero_damage":
                reward_struct.cur_frame_value = main_hero["totalHurtToHero"] / 2e4  # 归一化 default:2e4
            # 承受英雄伤害
            elif reward_name == "hero_hurt":
                reward_struct.cur_frame_value = main_hero["totalBeHurtByHero"] / 2e4
            # 总输出
            elif reward_name == "total_damage":
                reward_struct.cur_frame_value = main_hero["totalHurt"] / 6e4    # default:6e4
            # 没有动作
            elif reward_name == "no_ops":
                if main_hero["actor_state"]["behav_mode"] == "State_Idle":
                    reward_struct.cur_frame_value = True
                else:
                    reward_struct.cur_frame_value = False
            # 英雄是否在草丛中
            elif reward_name == "in_grass":
                if main_hero["isInGrass"]:
                    reward_struct.cur_frame_value = True
                else:
                    reward_struct.cur_frame_value = False



    # Calculate the total amount of experience gained using agent level and current experience value
    # 用智能体等级和当前经验值，计算获得经验值的总量
    def calculate_exp_sum(self, this_hero_info):
        exp_sum = 0.0
        for i in range(1, this_hero_info["level"]):
            exp_sum += self.m_each_level_max_exp[i]
        exp_sum += this_hero_info["exp"]
        return exp_sum

    # Calculate the forward reward based on the distance between the agent and both defensive towers
    # 用智能体到双方防御塔的距离，计算前进奖励
    def calculate_forward(self, main_hero, main_tower, enemy_tower):
        main_tower_pos = (main_tower["location"]["x"], main_tower["location"]["z"])
        enemy_tower_pos = (enemy_tower["location"]["x"], enemy_tower["location"]["z"])
        hero_pos = (
            main_hero["actor_state"]["location"]["x"],
            main_hero["actor_state"]["location"]["z"],
        )
        forward_value = 0
        dist_hero2emy = math.dist(hero_pos, enemy_tower_pos)
        dist_main2emy = math.dist(main_tower_pos, enemy_tower_pos)
        if main_hero["actor_state"]["hp"] / main_hero["actor_state"]["max_hp"] > 0.99 and dist_hero2emy > dist_main2emy:
            forward_value = (dist_main2emy - dist_hero2emy) / dist_main2emy
        return forward_value

    # Calculate the reward item information for both sides using frame data
    # 用帧数据来计算两边的奖励子项信息
    def frame_data_process(self, frame_data):
        main_camp, enemy_camp = -1, -1

        for hero in frame_data["hero_states"]:
            if hero["player_id"] == self.main_hero_player_id:
                main_camp = hero["actor_state"]["camp"]
                self.main_hero_camp = main_camp
            else:
                enemy_camp = hero["actor_state"]["camp"]
        self.set_cur_calc_frame_vec(self.m_main_calc_frame_map, frame_data, main_camp)
        self.set_cur_calc_frame_vec(self.m_enemy_calc_frame_map, frame_data, enemy_camp)

    # Use the values obtained in each frame to calculate the corresponding reward value
    # 用每一帧得到的奖励子项信息来计算对应的奖励值
    def get_reward(self, frame_data, reward_dict):
        reward_dict.clear()
        reward_sum, weight_sum = 0.0, 0.0
        for reward_name, reward_struct in self.m_cur_calc_frame_map.items():
            if reward_name == "hp_point":
                # 如果己方血量比敌方下降得更少，则带来正收益；反之则负收益。（零和模式）
                if (
                    self.m_main_calc_frame_map[reward_name].last_frame_value == 0.0
                    and self.m_enemy_calc_frame_map[reward_name].last_frame_value == 0.0
                ):
                    reward_struct.cur_frame_value = 0
                    reward_struct.last_frame_value = 0
                elif self.m_main_calc_frame_map[reward_name].last_frame_value == 0.0:
                    reward_struct.cur_frame_value = 0 - self.m_enemy_calc_frame_map[reward_name].cur_frame_value
                    reward_struct.last_frame_value = 0 - self.m_enemy_calc_frame_map[reward_name].last_frame_value
                elif self.m_enemy_calc_frame_map[reward_name].last_frame_value == 0.0:
                    reward_struct.cur_frame_value = self.m_main_calc_frame_map[reward_name].cur_frame_value - 0
                    reward_struct.last_frame_value = self.m_main_calc_frame_map[reward_name].last_frame_value - 0
                else:
                    reward_struct.cur_frame_value = (
                        self.m_main_calc_frame_map[reward_name].cur_frame_value
                        - self.m_enemy_calc_frame_map[reward_name].cur_frame_value
                    )
                    reward_struct.last_frame_value = (
                        self.m_main_calc_frame_map[reward_name].last_frame_value
                        - self.m_enemy_calc_frame_map[reward_name].last_frame_value
                    )
                reward_struct.value = reward_struct.cur_frame_value - reward_struct.last_frame_value
            elif reward_name == "ep_rate":
                reward_struct.cur_frame_value = self.m_main_calc_frame_map[reward_name].cur_frame_value
                reward_struct.last_frame_value = self.m_main_calc_frame_map[reward_name].last_frame_value
                if reward_struct.last_frame_value > 0:
                    reward_struct.value = reward_struct.cur_frame_value - reward_struct.last_frame_value
                else:
                    reward_struct.value = 0
            elif reward_name == "exp":
                main_hero = None
                for hero in frame_data["hero_states"]:
                    if hero["player_id"] == self.main_hero_player_id:
                        main_hero = hero
                if main_hero and main_hero["level"] >= 15:
                    reward_struct.value = 0
                else:
                    reward_struct.cur_frame_value = (
                        self.m_main_calc_frame_map[reward_name].cur_frame_value
                        - self.m_enemy_calc_frame_map[reward_name].cur_frame_value
                    )
                    reward_struct.last_frame_value = (
                        self.m_main_calc_frame_map[reward_name].last_frame_value
                        - self.m_enemy_calc_frame_map[reward_name].last_frame_value
                    )
                    reward_struct.value = reward_struct.cur_frame_value - reward_struct.last_frame_value
            elif reward_name == "forward":
                reward_struct.value = self.m_main_calc_frame_map[reward_name].cur_frame_value
            elif reward_name == "last_hit":
                reward_struct.value = self.m_main_calc_frame_map[reward_name].cur_frame_value
            elif reward_name == "no_ops":
                # “无操作”惩罚
                reward_struct.value = 1.0 if self.m_main_calc_frame_map["no_ops"].cur_frame_value else 0.0
            elif reward_name == "in_grass":
                # 蹲草当“老六”
                if not self.m_main_calc_frame_map["in_grass"].cur_frame_value:
                    reward_struct.value = 0.0
                else:
                    # 基础奖励：在草丛中 0.25
                    val = 0.25

                    # 用 next(...) 快速找到主英雄和敌英雄
                    main_hero = next(h for h in frame_data["hero_states"]
                                    if h["player_id"] == self.main_hero_player_id)
                    enemy_hero = next(h for h in frame_data["hero_states"]
                                    if h["player_id"] != self.main_hero_player_id)

                    # 1) “偷袭可见性” 条件：自己不可见且敌人可见 -> +0.5
                    # 可见阵营，camp_visible[0]表示是否蓝方可见，camp_visible[1]表示是否红方可见
                    main_vis_all = all(main_hero["actor_state"]["camp_visible"])
                    enemy_vis_all = all(enemy_hero["actor_state"]["camp_visible"])
                    if not main_vis_all and enemy_vis_all:
                        val += 0.5

                    # 2) 计算距离并判断是否在攻击范围内 -> +0.5
                    mx, mz = main_hero["actor_state"]["location"]['x'], main_hero["actor_state"]["location"]['z']
                    ex, ez = enemy_hero["actor_state"]["location"]['x'], enemy_hero["actor_state"]["location"]['z']
                    hero_dist = math.dist((mx, mz), (ex, ez))
                    attack_range = main_hero["actor_state"]['attack_range']

                    if hero_dist <= attack_range:
                        val += 0.5

                    reward_struct.value = val
            else:
                # Calculate zero-sum reward
                # 计算零和奖励
                reward_struct.cur_frame_value = (
                    self.m_main_calc_frame_map[reward_name].cur_frame_value
                    - self.m_enemy_calc_frame_map[reward_name].cur_frame_value
                )
                reward_struct.last_frame_value = (
                    self.m_main_calc_frame_map[reward_name].last_frame_value
                    - self.m_enemy_calc_frame_map[reward_name].last_frame_value
                )
                reward_struct.value = reward_struct.cur_frame_value - reward_struct.last_frame_value

            weight_sum += reward_struct.weight
            reward_sum += reward_struct.value * reward_struct.weight
            # reward_dict[reward_name] = reward_struct.value
            reward_dict[reward_name] = reward_struct.value * reward_struct.weight
        # reward_dict["reward_sum"] = reward_sum
        # 由于要局内衰减奖励，改为在result函数中计算总奖励

