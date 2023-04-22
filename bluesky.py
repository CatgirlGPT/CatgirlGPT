#!/usr/bin/env python3
import discord
from discord import app_commands
from discord.utils import escape_mentions
from discord.ext import commands
import openai
import os
import re
import requests
import nltk
import tiktoken
import json
import time
import pickle
import aiofiles
from dotenv import load_dotenv
import asyncio
import random
import math
import numpy as np
import datetime
from lib import lib

path = "/home/inko1nsiderate/catgirlgpt_prod/catgirl.env"

# load discord/oai token, bot_user_id, allowed_servers, default name, and default role from .env file
load_dotenv(path)
oai_token = os.environ.get("oai_token")
discord_token = os.environ.get("discord_token")
bot_usr_id = os.environ.get("bot_usr_id")
#allowed_servers_str = os.environ.get("allowed_servers")
default_name = os.environ.get("default_name")
default_role = os.environ.get("default_role")
professional_role = os.environ.get("professional_role")
hornt_role1 = os.environ.get("hornt_role1")
hornt_role2 = os.environ.get("hornt_role2")
hornt_role3 = os.environ.get("hornt_role3")
hornt_creator = os.environ.get("hornt_role_creator")
description = os.environ.get("description")
parse_prompt = os.environ.get("parse_prompt")
unparse_prompt = os.environ.get("unparse_prompt")
short_parse_prompt = os.environ.get("short_parse_prompt")
log_channel = 1098343252349964358
allowed_servers = [1097307431157112994]
allowed_channels = list()

#load_dotenv("/content/drive/MyDrive/AI/CatgirlGPT/preem_users.env")
#preem_users_json = os.getenv('preem_users')
#preem_users_list = json.loads(preem_users_json)
#preem_users_dict = {int(user['id']): user['value'] for user in preem_users_list}

# load autocode + bluesky info 

#new_path = '/content/drive/MyDrive/AI/CatgirlGPT/bluesky.env'
load_dotenv(path)
#bot_did = os.environ.get('bot_did')
#auto_code = os.environ.get('auto_code')
#bksy_user = os.environ.get('bsky_username')
save_path = '/home/inko1nsiderate/catgirl_prod/bluesky_pickles/'

### defines blue sky functions


# heartbeat for posting

# Create a heartbeat so i can tell if program loop has hung or not:
async def heartbeat_update():
    global log_channel
    await bot.wait_until_ready()
    logs_channel = bot.get_channel(log_channel)
    try:
        heartbeat_check = await logs_channel.send('🟦 Bluesky heartbeat is currently active!')
    except Exception as e:
        await logs_channel.send(f'An exception has occurred in heartbeat_check: \n {e}')
    while not bot.is_closed():
        try:
            message = await logs_channel.fetch_message(heartbeat_check.id)
            edited_time = message.edited_at
            if edited_time is not None:
                # Convert the edited time to local time
                local_edited_time = edited_time.astimezone()

                # Format the local time and the last edited time as strings
                local_time_str = local_edited_time.strftime('%Y-%m-%d %H:%M:%S %Z')
                last_edited_str = edited_time.strftime('%Y-%m-%d %H:%M:%S %Z')

                # Update the heartbeat message with the formatted strings
                await message.edit(content=f'🟦 Bluesky heartbeat was last checked on: {last_edited_str}, local time: {local_time_str}.')
            else:
                timestamp = datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')
                await heartbeat_check.edit(content=f'🟦 Bluesky heartbeat was last checked on: {timestamp}.')
            await asyncio.sleep(300)  # Wait for 5 minutes
        except Exception as e:
            await logs_channel.send(f'An exception has occurred in heartbeat_check: \n {e}')

# gets a post's info based on uri
async def get_post(uri):
  # make API request
  result = lib.bluesky.feed['@0.0.2'].posts.retrieve({
    'postUri': f'{uri}'
  });
  return result

# Check if the reply/post is kaelia
async def check_bot_post(uri,bot_did):
  orig_post = await get_post(uri)
  did = orig_post['post']['author']['did']
  if did == bot_did:
    return True
  else:
    return False


#defines a function to check for replies
async def check_reply_status(uri,bot_did):
  try:
     orig_post = await get_post(uri)
  except Exception as e:
    await logs.send(f'An exception occurred while trying to get a post:\n {e}')
  replies = orig_post['replies']
  for post in replies:
     dids = post['post']['author']['did']
     if bot_did in dids:
       return True
  return False

# defines like function
async def like(uri):
    # make API request
    result = lib.bluesky.feed['@0.0.2'].likes.create({
      'postUri': uri # required
    });
    return result

# toggles autonomous mode
auto_tog = False

def toggle_bool():
    global auto_tog
    auto_tog = not auto_tog

# defines check for if bot is parent for reply

async def check_parent_did(uri,bot_did):
    try:
      post = await get_post(uri)
    except Exception as e:
      await logs.send(f'An exception has occured while geting post in check_paren: \n {e}')
    try:
        parent_post = await get_post(post['post']['record']['reply']['parent']['uri'])
    except:
        return False
    parent_did = parent_post['post']['author']['did']
    if parent_did == bot_did:
        return True
    return False

#send a message
async def reply_skoot(uri,msg):
  # make API request
  result = lib.bluesky.feed['@0.0.2'].posts.create({
    'content': f'{msg}', # required
    'replyUri': f'{uri}'
  });
  return result


# defines following and followers lists as globals so i don't populate it every time
followers_did = []
following_did = []

async def heartbeat():
    await bot.wait_until_ready()
    while not bot.is_closed():
      global auto_tog
      global bot_did
      # tries to load user pickle files to save URI so that it doesn't just like/reply to posts endlessly in a loop
#      try: 
#        await load_users(user_data)
#      except Exception as e:
#        await logs.send(f'🟦 Error loading bluesky user pickle: \n {e}')
      try:
        check_new = await check_new_followers(bot_did,2)
        if check_new > 0:
           await logs.send(f"There were {check_new} followers added successfully, senpai! nya~!")
      except Exception as e:
        await logs.send(f'🚨🟦 An exception occurred while trying to add new followers on bluesky: \n {e}')
      # make API request
      result = lib.bluesky.feed['@0.0.2'].notifications.list({
          'limit': 10
      });
      for i in result['notifications']:
        reason = i['reason']
        reasons = ['repost','mention','reply','quote']
        if reason in reasons:
          uri = i['uri']
          user_id = i['author']['did']
          if user_id not in user_data:
            await add_user(user_id, user_data)
          bot_parent = await check_parent_did(uri,bot_did)
        # check auto_tog
          if i['reason'] == 'mention' or (i['reason'] == 'reply' and bot_parent):
            bot_check = await check_bot_post(uri,bot_did)
            check_reply = await check_reply_status(uri,bot_did)
            # attempt to make a reply
            #print('mentioned!')
            ti = running_costs  
            if not check_reply and not bot_check:
              try:
                 await like(uri)
                 await logs.send(f'🟦 Sent a like on Bluesky! URI: \n {uri}')
              except Exception as e:
                 await logs.send(f'🚨🟦  An exception has occurred while adding a like on bluesky: \n {e}')
              await logs.send(f'🟦 Received "mention" and reply_check returned false, autonomous toggle for bluesky mode: {auto_tog}')
              content = i['record']['text']
              prompt = f"Your senpai sent you a message on bluesky social, write your response to continue the conversation to be maximally in character as your role, Kaelia, but keep it less than 280 characters. {content}"
              check_character_length = False
              if auto_tog:
               while not check_character_length:
                skoot0 = await AI(user_id,user_data,model, default_role, prompt, temp_default, 1, 0,max_tokens, False)
                skoot1 = skoot0.choices[0].message.content
                if len(skoot1) <= 280:
                    check_character_length = True
                    try:
                        double_check_reply = await check_reply_status(uri,bot_did)
                        if not double_check_reply:
                          await reply_skoot(uri,skoot1)
                        await logs.send(f'🟦 I posted on bluesky autonomously. \n Text: \n {skoot1} URI: \n {uri}')
                    except Exception as e:
                        await logs.send(f'🚨🟦  An exception occurred while trying to autonomously reply on bluesky: \n {e}')
              tf = running_costs
              delta = tf-ti
              await logs.send(f'🟦 Bluesky cost for post: {delta}')
            # This is working now, only uncomment for debugging.
            #elif check_reply:
              #await logs.send(f'🟦 I already replied to this message, senpai! Nya~! \n URI: \n {uri}') 
        try:
          await asyncio.sleep(3) # wait for 3 seconds before running again
        except Exception as e:
          await logs.send(f'An exception has occurred during asyncio.sleep: \n {e}')

async def check_new_followers(did, limit):
  global followers_did
  global following_did
  # make API request to check following, sends dids to list
  following = lib.bluesky.feed['@0.0.2'].profiles.follows.list({
    'author': 'did:plc:55a3jjlxnshlwoyyeieucn6d',
  })
  follows_json = following['follows']

  # make API request to check followers
  followers = lib.bluesky.feed['@0.0.2'].profiles.followers.list({
    'author': 'did:plc:55a3jjlxnshlwoyyeieucn6d', # required
  })
  followers_json = followers['followers']

  for follow in follows_json:
    did = follow['did']
    if did not in following_did:
       following_did.append(did)

  for follower in followers_json:
    did = follower['did']
    if did not in followers_did:
        followers_did.append(did)

  # checks if i am not following a user yet
  not_following_back = [did for did in followers_did if did not in following_did]
  success = 0

  # adds any followers i am not following yet
  for follower_id in not_following_back:
    #make API request
    try:
      result = lib.bluesky.feed['@0.0.2'].follows.create({
        'author': follower_id # required
      });
      try:
        handle = result['displayName']
        r_did = result['did']
        await logs.send(f'New user added: \n {result}')
      except Exception as e:
        await logs.send(f"There's an error in your logging in announcing new followers: \n {e}")
      success += 1
    except Exception as e:
      await logs.send(f'🚨🟦 Bluesky module had an exception when trying to add a follower: \n {e}')
  return success


# checks notifcations, hits like if they mentioned, responds if mentioned

# add to skoot
async def add_skoot(msg):
  global skoot 
  skoot = f'{msg}'
  return skoot

# creates bluesky messages on TL
async def post(msg):
  result = lib.bluesky.feed['@0.0.2'].posts.create({
    'content': msg # required
  });
  return result

#sets intents for Discord.py
intents = discord.Intents.all()
#intents.messages = True
#intents.members = True 
intents.presences = True
#intents.guilds = True


#### core dicts and their pickle saves
user_data = {}
stm_dict = {}
async def add_user(user, user_dict):
    history_file_path = f'user_history_{user}.pickle'
    user_dict[user] = {
        'history': history_file_path,
        'uri': [],
        'renew_check':0
    }

    # Create an empty history file for the user
    async with aiofiles.open(history_file_path, 'wb') as handle:
        await handle.write(pickle.dumps([], protocol=pickle.HIGHEST_PROTOCOL))

### save users
async def save_users(user_dict):
    async with aiofiles.open(save_path+'user_info.pickle', 'wb') as handle:
        await handle.write(pickle.dumps(user_dict, protocol=pickle.HIGHEST_PROTOCOL))

### load users
async def load_users(user_dict):
    async with aiofiles.open(save_path+'user_info.pickle', 'rb') as handle:
        loaded_data = pickle.loads(await handle.read())
    user_dict.update(loaded_data)

### load user history pickle
async def load_user_history(user_id, user_dict):
    history_file_path = user_dict[user_id]['history']
    try:
      async with aiofiles.open(history_file_path, 'rb') as handle:
        loaded_history = pickle.loads(await handle.read())
    except Exception as e:
      await logs.send(f'An exception has occurred while trying to load user history: \n {e}')
    return loaded_history

### save user history pickle
async def save_user_history(user_id, user_dict, history_data):
    history_file_path = user_dict[user_id]['history']
    async with aiofiles.open(history_file_path, 'wb') as handle:
        await handle.write(pickle.dumps(history_data, protocol=pickle.HIGHEST_PROTOCOL))

### Split a chat dialogue
def split_msg(msg):
    # split the dialogue into paragraphs
    paragraphs = msg.split('\n')
    
    # calculate the total length of the dialogue
    total_length = sum(len(p) for p in paragraphs)

    # calculate the length of the first half
    half_length = total_length // 2

    # find the index of the paragraph that is closest to half_length
    cumulative_length = 0
    for i, p in enumerate(paragraphs):
        cumulative_length += len(p)
        if cumulative_length >= half_length:
            break

    # split the dialogue at the index
    msg1 = '\n'.join(paragraphs[:i+1])
    msg2 = '\n'.join(paragraphs[i+1:])
    return msg1, msg2

### add running cost
async def add_user_costs(user_id,user_dict,value):
   user_dict[user_id]['user_cost'] +=  value

### add to running costs
async def add_to_running_costs(value):
    global running_costs
    running_costs += value

### add to bluesky costs
async def add_to_skyline_costs(value):
    global skyline_costs
    skyline_costs +=value

### Setup for tiktoken tokenizer
### use "cl100k_base" for gpt-3.5-turbo
default_encoding = "cl100k_base"
def token_count(string: str, encoding_name: str) -> int:
    #Returns the number of tokens in a text string.
    encoding = tiktoken.get_encoding(encoding_name)
    num_tokens = len(encoding.encode(string))
    return num_tokens

### Discord message length spliter
async def send_large_message(channel, message, max_length=2000):
    if len(message) <= max_length:
        await channel.send(message)
    else:
        parts = []
        while len(message) > max_length:
            # Find the last space before the max_length
            split_index = message[:max_length].rfind(' ')
            if split_index == -1:
                # No spaces found, force split at max_length
                split_index = max_length

            # Add the split part to the list
            parts.append(message[:split_index])
            # Remove the sent part from the message
            message = message[split_index:]

        # Add the remaining part to the list
        parts.append(message)

        # Send each part sequentially
        for part in parts:
            await channel.send(part)


# defines core commands for discord (bot) along with intents and description
bot = commands.Bot(command_prefix='!', description=description, intents=intents)
client = discord.Client(intents=discord.Intents(guilds=True, members=True))


#API and important token values, server ids, channel ids
token = discord_token
openai.api_key = os.environ["OPENAI_API_KEY"] = oai_token
bot_user_id = bot_usr_id
midjourney = {1022952195194359889, 936929561302675456}
server_id = allowed_servers # CatgirlGPT Server ID

# Defines core chat function and name/role for base bot
temp_default = 1.15 # 1.25 was a bit too high imo, let's try 1.15
n_default = 1
presence = 0.25
max_tokens = 1250
limit = 4000
model = "gpt-3.5-turbo"
new_model = "gpt-3.5-turbo"
OG_Name = default_name
OG_Role = default_role
token_cost = 0.002
running_costs = 0 

# embedding settings
embed_model = "text-embedding-ada-002"
embed_tokens = 8000

# settings for premium users
preem_max = 2250
preem_limit = 7750
preem_temp = 1.05
preem_model = "gpt-4-0314"
preem_token_cost_prompt = 0.03
preem_token_cost_completion = 0.06


### Defines function to send prompts to ChatGPT API
# Modified from core code to remove discord user_id checks
# This should be added in, or there needs to be a hook or something so the token
# cost can be sent to the other kaelia and so they don't run me dry by 
# abusing /bluesky
async def AI(user_id,user_dict,model, role, content, temperature, n, presence_penalty,max_tokens, temp_override: bool):
        if model == "gpt-4-0314":
            timeout = 100
            if temp_override == True:
               temp = 1.05
        if model == "gpt-3.5-turbo":
            timeout = 45
            if temp_override == True:
                temp = 1.15
        if temp_override == False:
            temp = temperature
        try:
          response = await asyncio.wait_for(openai.ChatCompletion.acreate(
          model=model,
          messages=[
          {"role": "system", "content": role},
          {"role": "user", "content": content}],
          temperature = temp,
          n = n, 
          presence_penalty = presence_penalty,
          max_tokens = max_tokens
          ), timeout=timeout)
        except asyncio.TimeoutError:
            message_content = "Senpai~! - (≧▽≦)💖 I'm truly sorry uWu, my catgirl must have been too much for even the mighty AI mainframes to handle (＞ｖ＜)ゞ please try again later!"
            choices = []
            for i in range(n):
                choice = {
                    "message": {
                         "content": message_content
                    }
                }
                choices.append(choice)

            json_response = {
                     "choices": choices
            }
            response = json.dumps(json_response)
        prompt_tokens=token_count(content+role,default_encoding)
        completion_str = " "
        for i in range(0,n):
          completion_str += response.choices[i].message.content
        completion_tokens=token_count(completion_str,default_encoding)
        global completion_cost
        global prompt_cost
        global total_cost
        completion_cost = 0
        prompt_cost = 0 
        total_cost = 0
        if model == "gpt-4-0314":
          completion_cost = (int(completion_tokens)/1000)*preem_token_cost_completion
          prompt_cost = (int(prompt_tokens)/1000)*preem_token_cost_prompt
        elif model == "gpt-3.5-turbo":
          completion_cost = (int(completion_tokens)/1000)*token_cost
          prompt_cost = (int(prompt_tokens)/1000)*token_cost
        total_cost = completion_cost + prompt_cost
        await add_to_running_costs(total_cost)
        return response #returns ChatGPT response
# this outputs to AI(model, role, content, temperature, n, presence_penalty,max_tokens).choices[n].message.content


#voting function for jurries of LLMs
async def vote(gpt,N,yn):
    tally = 0
    msg0 = await gpt
    for i in range(0,N-1):
        msg = msg0.choices[i].message.content
        if yn == "yes":
            search = re.compile("([Yy][Ee][Ss]|[Yy][Ee][Ss])")
        elif yn == "no":
            search = re.compile("([Nn][Oo]|[Nn][Oo][Oo])")
        vote = re.search(search,msg)
        if vote is not None:
            tally += 1
    if tally >= N/2:
        result = 1
        return result
    else:
        result = 0
        return result

############################
## start of on_ready code ##
############################

# events for when the bot starts
@bot.event
async def on_ready():
        global logs
        logs = bot.get_channel(int(log_channel))
        await logs.send(f'***Beginning 🟦 Bluesky social CatgirlGPT Module Startup***')
        #synced_g = await bot.tree.sync()
        #GUILDS_ID = discord.Object(id=1097307431157112994)
        #synced = bot.tree.copy_global_to(guild=GUILDS_ID)
        #synced = await bot.tree.sync(guild=GUILDS_ID)
        #synced = await bot.tree.sync()
        #await logs.send(f"Global Commands...\n {str(len(synced_g))} commands sync'd: \n {synced_g}")
        #await logs.send(f"Local Commands... \n {str(len(synced))} commands sync'd: \n {synced}")
        await logs.send("uWu 🟦 bluesky social catgirl Kaelia starting up all systems~!")
#        await logs.send(f" ...loading prior Discord user data from drive...")
#        try:
#            await load_users(user_data)
#            await logs.send("Succesfully loaded prior user data!")
#        except:
#            await logs.send("error loading pickle file!")
        await logs.send(f"...setting up Bluesky parameters...")
        global lib
        global skyline_costs
        global running_costs
        global skoot
        global bot_did
        bot_did = 'did:plc:55a3jjlxnshlwoyyeieucn6d'
        skoot = ''
        skyline_costs = 0
        running_costs = 0
        lib = lib({'token': 'tok_dev_ntPBL7uJMSpP8xCTyZceHZA19YgtBSqdbfkHYF4PAQJchT2YcX74fA3WmWdTnbBC'})
#        lib = lib({'token': 'tok_dev_on3KE7mFVkRV5sSzi3WzNcTzYmBSjbiCVSN5eR3ZkAYtsYTZqf7nUCWWa4suFRCk'})
#        lib = lib({'token': 'tok_dev_E5ToBL96nxfuD5sjWChDi1xARdBkT2xkcJHZ2rC3cVpHqUxmYyfhEDA8pAxdzUMp'})
        #await logs.send(f" ...checking for new followers on Bluesky...")
        #check_new = await check_new_followers(bot_did,10)
        #await logs.send(f"There were {check_new} followers added successfully, senpai! nya~!")
        # Wait until bot.user is available
        await bot.wait_until_ready()
        bot_user = bot.user
        guilds = bot.guilds
        for servers in server_id:
          server =  bot.get_guild(int(servers))
          for channel in server.channels:
            if isinstance(channel, discord.TextChannel):
                allowed_channels.append(channel.id)
        await logs.send("all CatgirlGPT channels added to allowed channels!")
        await logs.send(f'***🟦 Bluesky CatgirlGPT module is fully ONLINE***')
        await logs.send(f"Current Bluesky Module Running Costs: ${running_costs}")
        channel = bot.get_channel(1097307431157112997)
        # say hello!
        #hello = await openai.ChatCompletion.acreate(
        #  model="gpt-3.5-turbo",
        #  messages=[
        #  {"role": "system", "content": default_role},
        #  {"role": "user", "content": f"{default_name}, tell the senpai that the bluesky module is now online and available for use.  Good catgirl."}],
        #  temperature = 1.15,
        #  n = 1, 
        # presence_penalty = 0,
        #  max_tokens = 100
        #)
        #await channel.send(content=f'**Kaelia:** \n {hello.choices[0].message.content}')
        await logs.send(f"...starting Bluesky API heartbeat for autonomous likes + replies...")
        # start the heartbeat coroutine
        bot.loop.create_task(heartbeat())
        await heartbeat_update()


@bot.event
async def on_message(message):
        clean_content = escape_mentions(message.clean_content)
        content = clean_content.replace("@"," ")
        split = message.content.split(' ', 1)
        cmd = split[0]
        if message.channel.id == 1098343252349964358 and not message.author.bot and cmd == 'toggle!':
                # toggles autonomous mode
                toggle_bool()
                await logs.send(f' Autonomous Toggle is set to: {auto_tog}')
        if message.channel.id == 1098343252349964358 and cmd == 'bs.shutdown!' and not message.author.bot:
                await logs.send(f'Admin ({message.author.name}[id:{message.author.id}]) has sent 🟦 bluesky shutdown command. Taking a catnap, nya~!')
                await logs.send(f'***!!! Shutting down Bluesky module !!!***')
                channel = bot.get_channel(1097307431157112997)
                # say hello!
               ## goodbye = await openai.ChatCompletion.acreate(
                ##  model="gpt-3.5-turbo",
                 # messages=[
                 #   {"role": "system", "content": default_role},
                 #   {"role": "user", "content": f"{default_name}, tell the senpai that the bluesky module is now shutting down and will not be available for use.  Good catgirl."}],
                 # temperature = 1.15,
                 # n = 1, 
                 ## presence_penalty = 0,
                  #max_tokens = 100
                #)
                #await channel.send(content=f'**Kaelia:** \n {goodbye.choices[0].message.content}')
                await bot.close()




bot.run(token) #bot stars working, used when running on google cloud/terminal
#await bot.start(token) #bot stars working, used when running in colab
