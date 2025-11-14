function love.load()
  love.window.setTitle("Pong in LÖVE")

  window_width, window_height = 800, 600
  love.window.setMode(window_width, window_height)

  paddle_width, paddle_height = 20, 100
  ball_size = 16

  paddle_speed = 300
  ball_speed = 200

  left_score, right_score = 0, 0

  ball_color = { r = 1, g = 1, b = 1 }

  left_paddle = {
    x = 40,
    y = window_height / 2 - paddle_height / 2
  }

  right_paddle = {
    x = window_width - 40 - paddle_width,
    y = window_height / 2 - paddle_height / 2
  }

  resetBall()
end

function resetBall()
  ball = {
    x = window_width / 2 - ball_size / 2,
    y = window_height / 2 - ball_size / 2,
    vx = ball_speed * (love.math.random(0, 1) == 0 and -1 or 1),
    vy = ball_speed * (love.math.random() * 2 - 1)
  }

  randomizeBallColor()
end

function randomizeBallColor()
  ball_color.r = love.math.random()
  ball_color.g = love.math.random()
  ball_color.b = love.math.random()
end

function love.update(dt)
  if love.keyboard.isDown("w") then
    left_paddle.y = left_paddle.y - paddle_speed * dt
  end
  if love.keyboard.isDown("s") then
    left_paddle.y = left_paddle.y + paddle_speed * dt
  end

  if love.keyboard.isDown("up") then
    right_paddle.y = right_paddle.y - paddle_speed * dt
  end
  if love.keyboard.isDown("down") then
    right_paddle.y = right_paddle.y + paddle_speed * dt
  end

  left_paddle.y = math.max(0, math.min(window_height - paddle_height, left_paddle.y))
  right_paddle.y = math.max(0, math.min(window_height - paddle_height, right_paddle.y))

  ball.x = ball.x + ball.vx * dt
  ball.y = ball.y + ball.vy * dt

  if ball.y <= 0 then
    ball.y = 0
    resetBall()
  elseif ball.y + ball_size >= window_height then
    ball.y = window_height - ball_size
    resetBall()
  end

  if checkCollision(ball.x, ball.y, ball_size, ball_size,
                     left_paddle.x, left_paddle.y, paddle_width, paddle_height) then
    resetBall()
  end

  if checkCollision(ball.x, ball.y, ball_size, ball_size,
                     right_paddle.x, right_paddle.y, paddle_width, paddle_height) then
    resetBall()
  end

  if ball.x + ball_size < 0 then
    right_score = right_score + 1
    resetBall()
  elseif ball.x > window_width then
    left_score = left_score + 1
    resetBall()
  end
end

function checkCollision(ax, ay, aw, ah, bx, by, bw, bh)
  return ax < bx + bw and
         bx < ax + aw and
         ay < by + bh and
         by < ay + ah
end

function love.draw()
  love.graphics.clear(0.05, 0.05, 0.1)

  love.graphics.setColor(1, 1, 1)

  love.graphics.rectangle("fill", window_width / 2 - 2, 0, 4, window_height)

  love.graphics.setFont(love.graphics.newFont(32))
  love.graphics.print(left_score, window_width / 4, 20)
  love.graphics.print(right_score, window_width * 3 / 4, 20)

  love.graphics.rectangle("fill", left_paddle.x, left_paddle.y,
                         paddle_width, paddle_height)
  love.graphics.rectangle("fill", right_paddle.x, right_paddle.y,
                         paddle_width, paddle_height)

  love.graphics.setColor(ball_color.r, ball_color.g, ball_color.b)
  love.graphics.rectangle("fill", ball.x, ball.y, ball_size, ball_size)

  love.graphics.setFont(love.graphics.newFont(14))
  love.graphics.print("W/S and Up/Down to move", 20, window_height - 30)
end
