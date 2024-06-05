# RoadEye
Trabalho de Visão Computacional - Detector de Infração de Trânsito

## Visão Geral

Bem-vindo ao projeto **RoadEye**! Este sistema inovador utiliza tecnologias avançadas de visão computacional e aprendizado de máquina para detectar automaticamente infrações de trânsito, como parar em local proibido e dirigir na contramão. Nosso objetivo é aumentar a segurança nas vias urbanas e reduzir o número de acidentes, proporcionando um ambiente mais seguro para motoristas e pedestres.

## Funcionalidades

### 1. Detecção de Parada em Local Proibido
- **Monitoramento Contínuo**: Câmeras instaladas em locais estratégicos capturam imagens em tempo real.
- **Análise Automática**: O sistema analisa as imagens para identificar veículos parados em áreas não permitidas.
- **Notificação Imediata**: Gera um alerta de infração no sistema.

### 2. Detecção de Direção na Contramão
- **Reconhecimento de Padrões**: Utiliza algoritmos de reconhecimento de padrões para identificar veículos que estão dirigindo na contramão.
- **Registro de Infrações**: Armazena os dados das infrações para análise posterior e tomada de decisão.
- **Notificação Imediata**: Gera um alerta de infração no sistema.

## Tecnologias Utilizadas

- **Visão Computacional**: Utiliza bibliotecas como YoloV4 para processamento de imagens.
- **Aprendizado de Máquina**: Implementação de modelos treinados em frameworks como PyTorch para reconhecimento de veículos e padrões de direção.

## Como Funciona

1. **Coleta de Dados**: Câmeras capturam vídeos das vias em tempo real.
2. **Processamento de Imagens**: As imagens são processadas para detectar veículos e analisar seu comportamento.
3. **Classificação de Infrações**: Algoritmos de aprendizado de máquina classificam as ações dos veículos como regulares ou infrações.
4. **Notificação e Armazenamento**: Infrações detectadas geram notificações no sistema e são registradas no banco de dados para futuras referências.

## Como Configurar

### Pré-requisitos

- Python 3.12
- PyTorch
- Streamlit (para interface web)
- Yolo4Tinyt

### Instalação

1. Clone o repositório:
    ```bash
    git clone https://github.com/GustavoMCF/RoadEye.git
    ```

2. Instale as dependências:
    ```bash
    Instalação no TERMINAL:
    pip install -r requirements.txt
    
    ```
    
    ```bash
    Instalação no VSCODE:
    ctrl+shift+p > Select Python Interpreter > Create Virtual Environment > .venv > Python 3.12 > requirements.txt
    
    ```


4. Inicie o servidor:
    ```bash
    streamlit run app.py
    ```

5. Acesse a interface web em `http://localhost:8081`.

## Contribuição

Sinta-se à vontade para contribuir com o projeto! Aqui estão algumas maneiras de ajudar:

- Reporte bugs e sugira novas funcionalidades através de Issues.
- Envie Pull Requests com melhorias e correções.
- Participe das discussões para desenvolver novas ideias e soluções.

## Licença

Este projeto está licenciado sob a licença MIT. Veja o arquivo `LICENSE` para mais detalhes.

Esperamos que este projeto seja útil e contribua para a melhoria da segurança no trânsito. Agradecemos pelo seu interesse e colaboração!

## Autores

- [Alexsandro Pasinato](https://github.com/Alekk123)
- [Diego Felix](https://github.com/Diegofelix1989)
- [Gustavo Ferreira](https://github.com/GustavoMCF)
- [Mirelly da Silva](https://github.com/MirellySilva)
- [Pedro Melo](https://github.com/PedroHenriqueMM)
- [Roberto Martins](https://github.com/Robertogithu)

---

Obrigado por apoiar o **RoadEye**! 🚗🚦